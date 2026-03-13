"""Tests for the ReAct loop engine."""

from __future__ import annotations

import pytest

from fangbot.brain.react import ReActLoop
from fangbot.memory.audit import AuditLogger, EventType
from fangbot.memory.session import SessionContext
from fangbot.models import ProviderResponse, ToolCall

from tests.conftest import MockMCPClient, MockProvider


@pytest.fixture
def audit_logger(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    logger.start_session()
    return logger


@pytest.fixture
def session():
    return SessionContext(system_prompt="Test system prompt")


class TestReActLoop:
    @pytest.mark.asyncio
    async def test_simple_tool_call_flow(self, sample_tools, audit_logger, session):
        """Agent calls a tool, then synthesizes — the happy path."""
        provider = MockProvider(
            responses=[
                # First response: LLM wants to call search
                ProviderResponse(
                    content="Let me search for the calculator.",
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            name="search_clinical_calculators",
                            arguments={"query": "CHA2DS2-VASc"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                # Second response: LLM wants to execute
                ProviderResponse(
                    content="Now executing the calculator.",
                    tool_calls=[
                        ToolCall(
                            id="tc_2",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "chadsvasc", "parameters": {"age": 72}},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                # Third response: final synthesis
                ProviderResponse(
                    content="The CHA2DS2-VASc score is 3.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient(
            tool_results={
                "search_clinical_calculators": "Found: CHA2DS2-VASc calculator (id: chadsvasc)",
                "execute_clinical_calculator": "Score: 3. DOI: 10.1234/example",
            }
        )

        loop = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit_logger)
        result = await loop.run("Calculate CHA2DS2-VASc for a 72yo male", session, sample_tools)

        assert result.synthesis == "The CHA2DS2-VASc score is 3."
        assert "search_clinical_calculators" in result.tool_calls_made
        assert "execute_clinical_calculator" in result.tool_calls_made
        assert result.guardrail_passed is True
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_guardrail_fires_on_no_tool_use(self, sample_tools, audit_logger, session):
        """Agent answers without tools — guardrail should flag violation."""
        provider = MockProvider(
            responses=[
                # LLM tries to answer directly without tools
                ProviderResponse(
                    content="The CHA2DS2-VASc score is 3 based on age and hypertension.",
                    stop_reason="end_turn",
                ),
                # After corrective injection, still no tools
                ProviderResponse(
                    content="I apologize, but I cannot access tools right now.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()
        loop = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit_logger)
        result = await loop.run("Calculate CHA2DS2-VASc", session, sample_tools)

        assert result.guardrail_passed is False
        assert len(result.guardrail_violations) > 0
        assert "No MCP tools were called" in result.guardrail_violations[0]

    @pytest.mark.asyncio
    async def test_max_iterations_safety_valve(self, sample_tools, audit_logger, session):
        """Agent that keeps calling tools should be stopped after max iterations."""
        infinite_tool_response = ProviderResponse(
            content="Searching again...",
            tool_calls=[
                ToolCall(
                    id="tc_loop",
                    name="search_clinical_calculators",
                    arguments={"query": "test"},
                )
            ],
            stop_reason="tool_use",
        )

        provider = MockProvider(responses=[infinite_tool_response] * 15)
        mcp = MockMCPClient()

        loop = ReActLoop(
            provider=provider, mcp_client=mcp, audit_logger=audit_logger, max_iterations=3
        )
        result = await loop.run("Test", session, sample_tools)

        assert result.iterations == 3
        assert "unable to complete" in result.synthesis.lower()

    @pytest.mark.asyncio
    async def test_tool_error_handled_gracefully(self, sample_tools, audit_logger, session):
        """MCP tool errors should be passed back to the LLM, not crash."""
        from fangbot.skills.mcp_client import MCPToolError

        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Executing calculator.",
                    tool_calls=[
                        ToolCall(
                            id="tc_err",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "bad"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="The calculator returned an error. Unable to compute.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()
        # Override call_tool to raise
        original_call = mcp.call_tool

        async def failing_call(name, args):
            if name == "execute_clinical_calculator":
                raise MCPToolError("Invalid calculator_id: bad")
            return await original_call(name, args)

        mcp.call_tool = failing_call

        loop = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit_logger)
        result = await loop.run("Test with bad calculator", session, sample_tools)

        # Should complete without raising
        assert result.synthesis is not None
        assert "execute_clinical_calculator" in result.tool_calls_made

    @pytest.mark.asyncio
    async def test_audit_events_recorded(self, sample_tools, audit_logger, session):
        """Verify all audit events are recorded during a run."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Searching.",
                    tool_calls=[
                        ToolCall(
                            id="tc_a",
                            name="search_clinical_calculators",
                            arguments={"query": "test"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Done. The result is X.",
                    stop_reason="end_turn",
                ),
            ]
        )
        mcp = MockMCPClient()

        loop = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit_logger)
        await loop.run("Test audit", session, sample_tools)

        events = audit_logger.get_events()
        event_types = [e.event_type for e in events]

        assert EventType.SESSION_START in event_types
        assert EventType.CASE_RECEIVED in event_types
        assert EventType.THINK in event_types
        assert EventType.TOOL_CALL in event_types
        assert EventType.TOOL_RESULT in event_types
        assert EventType.SYNTHESIS in event_types

    @pytest.mark.asyncio
    async def test_uncertainty_extracted_from_synthesis(self, sample_tools, audit_logger, session):
        """When synthesis contains an uncertainty block, it is parsed into result.uncertainty."""
        synthesis_with_block = (
            "The CHA2DS2-VASc score is 3.\n"
            "---\n"
            "Confidence: HIGH\n"
            "Reasoning: All parameters present and validated\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "---"
        )
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Searching.",
                    tool_calls=[
                        ToolCall(
                            id="tc_u1",
                            name="search_clinical_calculators",
                            arguments={"query": "CHA2DS2-VASc"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Executing.",
                    tool_calls=[
                        ToolCall(
                            id="tc_u2",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "chadsvasc", "parameters": {}},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content=synthesis_with_block,
                    stop_reason="end_turn",
                ),
            ]
        )
        mcp = MockMCPClient(
            tool_results={
                "search_clinical_calculators": "Found: CHA2DS2-VASc",
                "execute_clinical_calculator": "Score: 3",
            }
        )
        loop = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit_logger)
        result = await loop.run("Calculate CHA2DS2-VASc", session, sample_tools)

        assert result.uncertainty is not None
        assert result.uncertainty.confidence.value == "high"
        assert result.uncertainty.escalation_recommended is False
        # Synthesis should have the block stripped
        assert "Confidence:" not in result.synthesis

    @pytest.mark.asyncio
    async def test_uncertainty_none_when_no_block(self, sample_tools, audit_logger, session):
        """When synthesis has no uncertainty block, result.uncertainty is None."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Searching.",
                    tool_calls=[
                        ToolCall(
                            id="tc_n1",
                            name="search_clinical_calculators",
                            arguments={"query": "test"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="The score is 3. No uncertainty block.",
                    stop_reason="end_turn",
                ),
            ]
        )
        mcp = MockMCPClient()
        loop = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit_logger)
        result = await loop.run("Test", session, sample_tools)

        assert result.uncertainty is None

    @pytest.mark.asyncio
    async def test_uncertainty_audit_event_logged(self, sample_tools, audit_logger, session):
        """When uncertainty is extracted, a confidence_assessment audit event is logged."""
        synthesis_with_block = (
            "Result.\n"
            "---\n"
            "Confidence: MODERATE\n"
            "Reasoning: Age estimated\n"
            "Missing data: Exact DOB\n"
            "Contradictions: None\n"
            "---"
        )
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Searching.",
                    tool_calls=[
                        ToolCall(
                            id="tc_a1",
                            name="search_clinical_calculators",
                            arguments={"query": "test"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Executing.",
                    tool_calls=[
                        ToolCall(
                            id="tc_a2",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "test", "parameters": {}},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content=synthesis_with_block,
                    stop_reason="end_turn",
                ),
            ]
        )
        mcp = MockMCPClient(
            tool_results={
                "search_clinical_calculators": "Found",
                "execute_clinical_calculator": "Result",
            }
        )
        loop = ReActLoop(provider=provider, mcp_client=mcp, audit_logger=audit_logger)
        await loop.run("Test", session, sample_tools)

        events = audit_logger.get_events()
        confidence_events = [e for e in events if e.event_type == EventType.CONFIDENCE_ASSESSMENT]
        assert len(confidence_events) == 1
        assert confidence_events[0].data["confidence"] == "moderate"
