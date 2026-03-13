"""Tests for run_workflow integration in the ReAct loop."""

from __future__ import annotations

import pytest

from fangbot.brain.react import INTERNAL_TOOLS, ReActLoop
from fangbot.memory.audit import AuditLogger
from fangbot.memory.session import SessionContext
from fangbot.models import ProviderResponse, ToolCall, ToolDefinition
from fangbot.workflows.admission_oneliner import AdmissionOneLiner
from fangbot.workflows.engine import WorkflowEngine


@pytest.fixture
def workflow_engine() -> WorkflowEngine:
    engine = WorkflowEngine()
    engine.register(AdmissionOneLiner)
    return engine


class TestWorkflowReactIntegration:
    def test_run_workflow_in_internal_tools(self):
        assert "run_workflow" in INTERNAL_TOOLS

    @pytest.mark.asyncio
    async def test_run_workflow_routed_to_engine(
        self, tmp_path, mock_provider, mock_mcp, workflow_engine
    ):
        # First call: LLM decides to call run_workflow
        mock_provider.queue_response(
            ProviderResponse(
                content="I'll generate an admission one-liner.",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="run_workflow",
                        arguments={
                            "workflow_name": "admission_oneliner",
                            "clinical_text": "72M with CHF presenting with SOB.",
                        },
                    )
                ],
                stop_reason="tool_use",
            )
        )
        # Second call (for workflow's LLM generation step)
        mock_provider.queue_response(
            ProviderResponse(
                content="72M with CHF presenting with shortness of breath.",
                stop_reason="end_turn",
            )
        )
        # Third call: final synthesis after tool result
        mock_provider.queue_response(
            ProviderResponse(
                content="Here is the admission one-liner draft.",
                stop_reason="end_turn",
            )
        )

        audit = AuditLogger(log_dir=str(tmp_path))
        audit.start_session()

        react = ReActLoop(
            provider=mock_provider,
            mcp_client=mock_mcp,
            audit_logger=audit,
            workflow_engine=workflow_engine,
        )

        session = SessionContext(system_prompt="Test")
        tools = [
            ToolDefinition(
                name="run_workflow",
                description="Run a workflow",
                input_schema={"type": "object", "properties": {}},
            )
        ]

        result = await react.run("Generate admission one-liner", session, tools)

        assert "run_workflow" in result.tool_calls_made
        # Verify the workflow was executed (tool result was added to session)
        tool_results = [m for m in session.messages if m.role.value == "tool"]
        assert len(tool_results) >= 1

    @pytest.mark.asyncio
    async def test_run_workflow_without_engine_returns_error(
        self, tmp_path, mock_provider, mock_mcp
    ):
        mock_provider.queue_response(
            ProviderResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="run_workflow",
                        arguments={
                            "workflow_name": "test",
                            "clinical_text": "data",
                        },
                    )
                ],
                stop_reason="tool_use",
            )
        )
        mock_provider.queue_response(
            ProviderResponse(content="Error noted.", stop_reason="end_turn")
        )

        audit = AuditLogger(log_dir=str(tmp_path))
        audit.start_session()

        react = ReActLoop(
            provider=mock_provider, mcp_client=mock_mcp, audit_logger=audit
        )
        session = SessionContext(system_prompt="Test")

        result = await react.run("test", session, [])
        # Should return an error about missing engine
        tool_messages = [m for m in session.messages if m.role.value == "tool"]
        assert any("ERROR" in (m.content or "") for m in tool_messages)
