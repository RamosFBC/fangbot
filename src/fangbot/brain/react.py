"""ReAct loop engine — the core reasoning + acting cycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fangbot.brain.guardrails import GuardrailResult, run_all_guardrails
from fangbot.brain.progress import NullProgress, ProgressCallback
from fangbot.brain.providers.base import LLMProvider
from fangbot.memory.audit import AuditLogger, EventType
from fangbot.memory.session import SessionContext
from fangbot.models import ToolCall, ToolDefinition
from fangbot.skills.mcp_client import MCPToolError, OpenMedicineMCPClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10

# Internal tools handled by the ReAct loop, not forwarded to MCP
INTERNAL_TOOLS = {"load_clinical_skill"}


@dataclass
class ReActResult:
    synthesis: str
    tool_calls_made: list[str] = field(default_factory=list)
    chain_of_thought: list[str] = field(default_factory=list)
    iterations: int = 0
    guardrail_violations: list[str] = field(default_factory=list)
    guardrail_passed: bool = True


class ReActLoop:
    """Core ReAct engine: reason, act (call tools), observe, repeat."""

    def __init__(
        self,
        provider: LLMProvider,
        mcp_client: OpenMedicineMCPClient,
        audit_logger: AuditLogger,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        clinical_skill_loader: object | None = None,
    ):
        self._provider = provider
        self._mcp = mcp_client
        self._audit = audit_logger
        self._max_iterations = max_iterations
        self._skill_loader = clinical_skill_loader

    async def run(
        self,
        user_input: str,
        session: SessionContext,
        tools: list[ToolDefinition],
        progress: ProgressCallback | None = None,
    ) -> ReActResult:
        """Execute the ReAct loop for a single user query."""
        cb = progress or NullProgress()
        result = ReActResult(synthesis="")

        self._audit.log(EventType.CASE_RECEIVED, {"input": user_input})
        session.add_user_message(user_input)

        for iteration in range(1, self._max_iterations + 1):
            result.iterations = iteration
            cb.on_iteration(iteration, self._max_iterations)
            logger.debug(f"ReAct iteration {iteration}")

            try:
                response = await self._provider.call(
                    messages=session.messages,
                    tools=tools,
                    system=session.system_prompt,
                )
            except Exception as e:
                error_msg = f"LLM API error: {e}"
                logger.error(error_msg)
                self._audit.log(EventType.TOOL_ERROR, {"error": error_msg})
                result.synthesis = f"Error calling {self._provider.model_name}: {e}"
                result.guardrail_passed = False
                result.guardrail_violations = [error_msg]
                return result

            # Record thinking
            if response.content:
                result.chain_of_thought.append(response.content)
                self._audit.log_think(response.content)
                cb.on_thinking(response.content)

            # If no tool calls, this is the final response
            if not response.tool_calls:
                result.synthesis = response.content

                # Run guardrails
                guardrail = run_all_guardrails(result.tool_calls_made)
                if not guardrail.passed and guardrail.corrective_message:
                    cb.on_guardrail_correction(guardrail.violations)
                    corrective_result = await self._try_corrective(
                        session, tools, guardrail, result, cb
                    )
                    if corrective_result is not None:
                        return corrective_result

                result.guardrail_passed = guardrail.passed
                result.guardrail_violations = guardrail.violations
                if guardrail.violations:
                    for v in guardrail.violations:
                        self._audit.log(EventType.GUARDRAIL_VIOLATION, {"violation": v})

                self._audit.log_synthesis(result.synthesis)
                session.add_assistant_message(result.synthesis)
                return result

            # Execute tool calls
            session.add_assistant_message(response.content, tool_calls=response.tool_calls)
            await self._execute_tool_calls(response.tool_calls, session, result, cb)

        # Max iterations reached
        result.synthesis = (
            "I was unable to complete the analysis within the allowed number of steps. "
            "Please try rephrasing your question or providing more specific information."
        )
        self._audit.log_synthesis(result.synthesis)
        session.add_assistant_message(result.synthesis)
        return result

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        session: SessionContext,
        result: ReActResult,
        cb: ProgressCallback | None = None,
    ) -> None:
        """Execute a batch of tool calls, routing internal vs. MCP tools."""
        _cb = cb or NullProgress()
        for tc in tool_calls:
            self._audit.log_tool_call(tc.name, tc.arguments)
            session.record_tool_call(tc.name)
            result.tool_calls_made.append(tc.name)
            _cb.on_tool_start(tc.name, tc.arguments)

            if tc.name in INTERNAL_TOOLS:
                # Handle internal tool
                tool_output = self._handle_internal_tool(tc)
                self._audit.log_tool_result(tc.name, tool_output)
                session.add_tool_result(tc.id, tool_output)
                _cb.on_tool_result(tc.name, tool_output)
            else:
                # Forward to MCP
                try:
                    tool_output = await self._mcp.call_tool(tc.name, tc.arguments)
                    self._audit.log_tool_result(tc.name, tool_output)
                    session.add_tool_result(tc.id, tool_output)
                    _cb.on_tool_result(tc.name, tool_output)
                except MCPToolError as e:
                    error_msg = str(e)
                    self._audit.log_tool_error(tc.name, error_msg)
                    session.add_tool_result(tc.id, f"ERROR: {error_msg}")
                    _cb.on_tool_result(tc.name, error_msg, is_error=True)

    def _handle_internal_tool(self, tc: ToolCall) -> str:
        """Handle an internal tool call (not forwarded to MCP)."""
        if tc.name == "load_clinical_skill":
            return self._handle_load_clinical_skill(tc.arguments)
        return f"ERROR: Unknown internal tool: {tc.name}"

    def _handle_load_clinical_skill(self, arguments: dict) -> str:
        """Load a clinical skill and return its content."""
        if self._skill_loader is None:
            return "ERROR: Clinical skill loader not configured."

        skill_name = arguments.get("skill_name", "")
        reason = arguments.get("reason", "")

        try:
            content = self._skill_loader.load_skill(skill_name)
            self._audit.log(
                EventType.SKILL_LOADED,
                {"skill_name": skill_name, "reason": reason},
            )
            logger.info(f"Loaded clinical skill: {skill_name} (reason: {reason})")
            return content
        except Exception as e:
            error_msg = f"Failed to load skill '{skill_name}': {e}"
            logger.warning(error_msg)
            return f"ERROR: {error_msg}"

    async def _try_corrective(
        self,
        session: SessionContext,
        tools: list[ToolDefinition],
        guardrail: GuardrailResult,
        result: ReActResult,
        cb: ProgressCallback | None = None,
    ) -> ReActResult | None:
        """Inject a corrective message and re-run if guardrails failed."""
        _cb = cb or NullProgress()
        logger.warning(f"Guardrail violations: {guardrail.violations}. Attempting correction.")

        # Add the LLM's response, then inject corrective user message
        session.add_assistant_message(result.synthesis)
        session.add_user_message(guardrail.corrective_message or "")

        self._audit.log(
            EventType.GUARDRAIL_VIOLATION,
            {
                "violations": guardrail.violations,
                "action": "corrective_injection",
            },
        )

        # One more pass with remaining iterations
        try:
            corrective_response = await self._provider.call(
                messages=session.messages,
                tools=tools,
                system=session.system_prompt,
            )
        except Exception as e:
            logger.error(f"LLM API error during corrective attempt: {e}")
            return None

        if corrective_response.tool_calls:
            # Good — the LLM is now trying to use tools
            session.add_assistant_message(
                corrective_response.content, tool_calls=corrective_response.tool_calls
            )
            await self._execute_tool_calls(corrective_response.tool_calls, session, result, _cb)

            # Continue the loop for remaining iterations
            for _ in range(self._max_iterations - result.iterations):
                result.iterations += 1
                _cb.on_iteration(result.iterations, self._max_iterations)
                try:
                    response = await self._provider.call(
                        messages=session.messages,
                        tools=tools,
                        system=session.system_prompt,
                    )
                except Exception:
                    return None
                if response.content:
                    result.chain_of_thought.append(response.content)
                    self._audit.log_think(response.content)
                    _cb.on_thinking(response.content)

                if not response.tool_calls:
                    result.synthesis = response.content
                    final_check = run_all_guardrails(result.tool_calls_made)
                    result.guardrail_passed = final_check.passed
                    result.guardrail_violations = final_check.violations
                    self._audit.log_synthesis(result.synthesis)
                    session.add_assistant_message(result.synthesis)
                    return result

                session.add_assistant_message(response.content, tool_calls=response.tool_calls)
                await self._execute_tool_calls(response.tool_calls, session, result, _cb)

        # Corrective attempt also failed — return None to use original result
        return None
