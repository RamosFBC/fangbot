"""ReAct loop engine — the core reasoning + acting cycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fangbot.brain.guardrails import GuardrailResult, run_all_guardrails
from fangbot.brain.progress import NullProgress, ProgressCallback
from fangbot.brain.providers.base import LLMProvider
from fangbot.brain.uncertainty import (
    UncertaintyAssessment,
    parse_uncertainty_assessment,
    strip_uncertainty_block,
)
from fangbot.memory.audit import AuditLogger, EventType
from fangbot.memory.session import SessionContext
from fangbot.models import ToolCall, ToolDefinition
from fangbot.skills.evidence import EvidenceTracker
from fangbot.skills.mcp_client import MCPToolError, OpenMedicineMCPClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 10

# Internal tools handled by the ReAct loop, not forwarded to MCP
INTERNAL_TOOLS = {"load_clinical_skill", "parse_patient_chart", "run_workflow"}


@dataclass
class ReActResult:
    synthesis: str
    tool_calls_made: list[str] = field(default_factory=list)
    chain_of_thought: list[str] = field(default_factory=list)
    iterations: int = 0
    guardrail_violations: list[str] = field(default_factory=list)
    guardrail_passed: bool = True
    uncertainty: UncertaintyAssessment | None = None


class ReActLoop:
    """Core ReAct engine: reason, act (call tools), observe, repeat."""

    def __init__(
        self,
        provider: LLMProvider,
        mcp_client: OpenMedicineMCPClient,
        audit_logger: AuditLogger,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        clinical_skill_loader: object | None = None,
        chart_parser: object | None = None,
        workflow_engine: object | None = None,
        evidence_tracker: EvidenceTracker | None = None,
    ):
        self._provider = provider
        self._mcp = mcp_client
        self._audit = audit_logger
        self._max_iterations = max_iterations
        self._skill_loader = clinical_skill_loader
        self._chart_parser = chart_parser
        self._workflow_engine = workflow_engine
        self._evidence_tracker = evidence_tracker

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

                self._extract_uncertainty(result)
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
        self._extract_uncertainty(result)
        self._audit.log_synthesis(result.synthesis)
        session.add_assistant_message(result.synthesis)
        return result

    def _extract_uncertainty(self, result: ReActResult) -> None:
        """Parse uncertainty block from synthesis and log audit events."""
        assessment = parse_uncertainty_assessment(result.synthesis)
        if assessment is None:
            return

        result.uncertainty = assessment
        result.synthesis = strip_uncertainty_block(result.synthesis)

        self._audit.log_confidence_assessment(
            confidence=assessment.confidence.value,
            reasoning=assessment.reasoning,
            missing_data=assessment.missing_data,
            contradictions=assessment.contradictions,
            escalation_recommended=assessment.escalation_recommended,
        )

    def _process_evidence(self, tool_name: str, result_text: str) -> None:
        """Extract and log evidence from guideline tool results."""
        if self._evidence_tracker is None:
            return

        prev_citations = len(self._evidence_tracker.citations)
        prev_guidelines = len(self._evidence_tracker.guidelines)

        self._evidence_tracker.process_tool_result(tool_name, result_text)

        # Log only NEW guidelines
        for ref in self._evidence_tracker.guidelines[prev_guidelines:]:
            self._audit.log_guideline_retrieved(
                guideline_id=ref.guideline_id,
                title=ref.title,
                organization=ref.organization,
                sections=ref.sections_consulted,
            )

        # Log only NEW citations
        for citation in self._evidence_tracker.citations[prev_citations:]:
            self._audit.log_evidence_cited(
                doi=citation.doi,
                pmid=citation.pmid,
                recommendation=citation.recommendation,
                source=citation.source.value,
                strength=citation.strength.value if citation.strength else None,
            )

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
                tool_output = await self._handle_internal_tool(tc)
                self._audit.log_tool_result(tc.name, tool_output)
                session.add_tool_result(tc.id, tool_output)
                _cb.on_tool_result(tc.name, tool_output)
            else:
                # Forward to MCP
                try:
                    tool_output = await self._mcp.call_tool(tc.name, tc.arguments)
                    self._audit.log_tool_result(tc.name, tool_output)
                    # Track evidence from guideline tools
                    if self._evidence_tracker is not None:
                        self._process_evidence(tc.name, tool_output)
                    session.add_tool_result(tc.id, tool_output)
                    _cb.on_tool_result(tc.name, tool_output)
                except MCPToolError as e:
                    error_msg = str(e)
                    self._audit.log_tool_error(tc.name, error_msg)
                    session.add_tool_result(tc.id, f"ERROR: {error_msg}")
                    _cb.on_tool_result(tc.name, error_msg, is_error=True)

    async def _handle_internal_tool(self, tc: ToolCall) -> str:
        """Handle an internal tool call (not forwarded to MCP)."""
        if tc.name == "load_clinical_skill":
            return self._handle_load_clinical_skill(tc.arguments)
        if tc.name == "parse_patient_chart":
            return await self._handle_parse_patient_chart(tc.arguments)
        if tc.name == "run_workflow":
            return await self._handle_run_workflow(tc.arguments)
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

    async def _handle_parse_patient_chart(self, arguments: dict) -> str:
        """Parse clinical text into structured chart data."""
        if self._chart_parser is None:
            return "ERROR: Chart parser not configured."

        clinical_text = arguments.get("clinical_text", "")
        if not clinical_text:
            return "ERROR: clinical_text is required."

        try:
            chart = await self._chart_parser.parse(clinical_text)
            self._audit.log(
                EventType.CHART_PARSE,
                {
                    "facts_count": len(chart.facts),
                    "warnings_count": len(chart.parse_warnings),
                    "categories": list({f.category.value for f in chart.facts}),
                },
            )
            logger.info(
                f"Chart parsed: {len(chart.facts)} facts, {len(chart.parse_warnings)} warnings"
            )
            return chart.model_dump_json(indent=2)
        except Exception as e:
            error_msg = f"Chart parsing failed: {e}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"

    async def _handle_run_workflow(self, arguments: dict) -> str:
        """Execute a clinical workflow and return the draft as JSON."""
        if self._workflow_engine is None:
            return "ERROR: Workflow engine not configured."

        workflow_name = arguments.get("workflow_name", "")
        clinical_text = arguments.get("clinical_text", "")
        if not workflow_name:
            return "ERROR: workflow_name is required."
        if not clinical_text:
            return "ERROR: clinical_text is required."

        try:
            from fangbot.chart.models import PatientChart
            from fangbot.workflows.engine import WorkflowContext

            chart = PatientChart(facts=[], raw_text=clinical_text, parse_warnings=[])
            context = WorkflowContext(
                chart=chart,
                provider=self._provider,
                audit=self._audit,
                raw_text=clinical_text,
            )
            draft = await self._workflow_engine.run(workflow_name, context)
            logger.info(f"Workflow '{workflow_name}' completed: {len(draft.sections)} sections")
            return draft.model_dump_json(indent=2)
        except KeyError as e:
            return f"ERROR: {e}"
        except Exception as e:
            error_msg = f"Workflow execution failed: {e}"
            logger.error(error_msg)
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
                    self._extract_uncertainty(result)
                    self._audit.log_synthesis(result.synthesis)
                    session.add_assistant_message(result.synthesis)
                    return result

                session.add_assistant_message(response.content, tool_calls=response.tool_calls)
                await self._execute_tool_calls(response.tool_calls, session, result, _cb)

        # Corrective attempt also failed — return None to use original result
        return None
