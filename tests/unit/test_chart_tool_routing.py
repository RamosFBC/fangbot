"""Tests for parse_patient_chart internal tool in the ReAct loop."""

from __future__ import annotations

import pytest

from fangbot.brain.react import INTERNAL_TOOLS, ReActLoop
from fangbot.chart.parser import ChartParser
from fangbot.memory.audit import AuditLogger, EventType
from fangbot.memory.session import SessionContext
from fangbot.models import ProviderResponse, ToolCall

from tests.conftest import MockMCPClient, MockProvider, SAMPLE_TOOLS


class ChartExtractionMockProvider(MockProvider):
    """Provider that returns chart extraction tool calls when used by ChartParser."""

    def __init__(self, extraction_data: dict, responses: list[ProviderResponse] | None = None):
        super().__init__(responses)
        self._extraction_data = extraction_data

    async def call(self, messages, tools=None, system=None):
        # If called with the extraction tool, return extraction data
        if tools and any(t.name == "submit_chart_extraction" for t in tools):
            return ProviderResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc_extract",
                        name="submit_chart_extraction",
                        arguments=self._extraction_data,
                    )
                ],
                stop_reason="tool_use",
            )
        # Otherwise use normal mock behavior
        return await super().call(messages, tools, system)


@pytest.fixture
def audit_logger(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    logger.start_session()
    return logger


@pytest.fixture
def session():
    return SessionContext(system_prompt="Test system prompt")


class TestParsePatientChartTool:
    def test_parse_patient_chart_is_internal_tool(self):
        assert "parse_patient_chart" in INTERNAL_TOOLS

    @pytest.mark.asyncio
    async def test_parse_chart_returns_structured_facts(self, audit_logger, session):
        extraction = {
            "facts": [
                {
                    "name": "Creatinine",
                    "value": "1.8 mg/dL",
                    "category": "lab",
                    "source": "BMP results",
                }
            ],
            "warnings": [],
        }
        provider = ChartExtractionMockProvider(
            extraction_data=extraction,
            responses=[
                # Agent calls parse_patient_chart
                ProviderResponse(
                    content="I'll parse this chart.",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="parse_patient_chart",
                            arguments={"clinical_text": "Cr 1.8 mg/dL on BMP"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                # Agent uses parsed data and calls MCP tools
                ProviderResponse(
                    content="Now searching for calculator.",
                    tool_calls=[
                        ToolCall(
                            id="tc2",
                            name="search_clinical_calculators",
                            arguments={"query": "CKD-EPI"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc3",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "ckd_epi", "parameters": {}},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Based on the chart, CKD-EPI GFR is calculated.",
                    stop_reason="end_turn",
                ),
            ],
        )

        mcp = MockMCPClient()
        chart_parser = ChartParser(provider)
        loop = ReActLoop(provider, mcp, audit_logger, chart_parser=chart_parser)

        result = await loop.run("Patient has Cr 1.8", session, SAMPLE_TOOLS)

        assert "parse_patient_chart" in result.tool_calls_made
        # MCP tools should also have been called
        mcp_names = [c["name"] for c in mcp.calls]
        assert "search_clinical_calculators" in mcp_names

    @pytest.mark.asyncio
    async def test_parse_chart_without_parser_returns_error(self, audit_logger, session):
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Let me parse.",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="parse_patient_chart",
                            arguments={"clinical_text": "some text"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Parser not available.",
                    tool_calls=[
                        ToolCall(
                            id="tc2",
                            name="search_clinical_calculators",
                            arguments={"query": "test"},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc3",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "test", "parameters": {}},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(content="Done.", stop_reason="end_turn"),
            ]
        )

        mcp = MockMCPClient()
        # No chart_parser passed
        loop = ReActLoop(provider, mcp, audit_logger)

        result = await loop.run("test", session, SAMPLE_TOOLS)

        assert "parse_patient_chart" in result.tool_calls_made
        # Check that error was returned in session
        tool_msgs = [m for m in session.messages if m.role.value == "tool"]
        assert any("not configured" in m.content.lower() for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_chart_parse_audit_event(self, audit_logger, session):
        extraction = {
            "facts": [{"name": "HR", "value": "88", "category": "vital", "source": "Vitals"}],
            "warnings": [],
        }
        provider = ChartExtractionMockProvider(
            extraction_data=extraction,
            responses=[
                ProviderResponse(
                    content="Parsing chart.",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="parse_patient_chart",
                            arguments={"clinical_text": "HR 88"},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc2",
                            name="search_clinical_calculators",
                            arguments={"query": "test"},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc3",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "t", "parameters": {}},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(content="Heart rate is 88.", stop_reason="end_turn"),
            ],
        )

        mcp = MockMCPClient()
        chart_parser = ChartParser(provider)
        loop = ReActLoop(provider, mcp, audit_logger, chart_parser=chart_parser)

        await loop.run("HR 88", session, SAMPLE_TOOLS)

        events = audit_logger.get_events()
        chart_events = [e for e in events if e.event_type == EventType.CHART_PARSE]
        assert len(chart_events) == 1
        assert chart_events[0].data["facts_count"] == 1

    @pytest.mark.asyncio
    async def test_parse_chart_not_sent_to_mcp(self, audit_logger, session):
        """parse_patient_chart should be intercepted, not forwarded to MCP."""
        extraction = {
            "facts": [{"name": "BP", "value": "140/90", "category": "vital", "source": "Vitals"}],
            "warnings": [],
        }
        provider = ChartExtractionMockProvider(
            extraction_data=extraction,
            responses=[
                ProviderResponse(
                    content="Parsing.",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="parse_patient_chart",
                            arguments={"clinical_text": "BP 140/90"},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc2",
                            name="search_clinical_calculators",
                            arguments={"query": "test"},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc3",
                            name="execute_clinical_calculator",
                            arguments={"calculator_id": "t", "parameters": {}},
                        ),
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(content="BP is elevated.", stop_reason="end_turn"),
            ],
        )

        mcp = MockMCPClient()
        chart_parser = ChartParser(provider)
        loop = ReActLoop(provider, mcp, audit_logger, chart_parser=chart_parser)

        await loop.run("BP check", session, SAMPLE_TOOLS)

        mcp_names = [c["name"] for c in mcp.calls]
        assert "parse_patient_chart" not in mcp_names
