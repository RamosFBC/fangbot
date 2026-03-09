"""ReAct loop engine — the core reasoning + acting cycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fangbot.brain.guardrails import GuardrailResult, run_all_guardrails
from fangbot.brain.providers.base import LLMProvider
from fangbot.memory.audit import AuditLogger, EventType
from fangbot.memory.session import SessionContext
from fangbot.models import ToolCall, ToolDefinition
from fangbot.skills.mcp_client import MCPToolError, OpenMedicineMCPClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10


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
    ):
        self._provider = provider
        self._mcp = mcp_client
        self._audit = audit_logger
        self._max_iterations = max_iterations

    async def run(
        self,
        user_input: str,
        session: SessionContext,
        tools: list[ToolDefinition],
    ) -> ReActResult:
        """Execute the ReAct loop for a single user query."""
        result = ReActResult(synthesis="")

        self._audit.log(EventType.CASE_RECEIVED, {"input": user_input})
        session.add_user_message(user_input)

        for iteration in range(1, self._max_iterations + 1):
            result.iterations = iteration
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

            # If no tool calls, this is the final response
            if not response.tool_calls:
                result.synthesis = response.content

                # Run guardrails
                guardrail = run_all_guardrails(result.tool_calls_made)
                if not guardrail.passed and guardrail.corrective_message:
                    # Give the LLM one corrective chance
                    corrective_result = await self._try_corrective(
                        session, tools, guardrail, result
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
            await self._execute_tool_calls(response.tool_calls, session, result)

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
    ) -> None:
        """Execute a batch of tool calls via MCP and record results."""
        for tc in tool_calls:
            self._audit.log_tool_call(tc.name, tc.arguments)
            session.record_tool_call(tc.name)
            result.tool_calls_made.append(tc.name)

            try:
                tool_output = await self._mcp.call_tool(tc.name, tc.arguments)
                self._audit.log_tool_result(tc.name, tool_output)
                session.add_tool_result(tc.id, tool_output)
            except MCPToolError as e:
                error_msg = str(e)
                self._audit.log_tool_error(tc.name, error_msg)
                session.add_tool_result(tc.id, f"ERROR: {error_msg}")

    async def _try_corrective(
        self,
        session: SessionContext,
        tools: list[ToolDefinition],
        guardrail: GuardrailResult,
        result: ReActResult,
    ) -> ReActResult | None:
        """Inject a corrective message and re-run if guardrails failed."""
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
            await self._execute_tool_calls(corrective_response.tool_calls, session, result)

            # Continue the loop for remaining iterations
            for _ in range(self._max_iterations - result.iterations):
                result.iterations += 1
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

                if not response.tool_calls:
                    result.synthesis = response.content
                    final_check = run_all_guardrails(result.tool_calls_made)
                    result.guardrail_passed = final_check.passed
                    result.guardrail_violations = final_check.violations
                    self._audit.log_synthesis(result.synthesis)
                    session.add_assistant_message(result.synthesis)
                    return result

                session.add_assistant_message(response.content, tool_calls=response.tool_calls)
                await self._execute_tool_calls(response.tool_calls, session, result)

        # Corrective attempt also failed — return None to use original result
        return None
