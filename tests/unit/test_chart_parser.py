"""Tests for LLM-assisted chart parser."""

from __future__ import annotations

import pytest

from fangbot.brain.providers.base import LLMProvider
from fangbot.chart.models import FactCategory, FactStatus, PatientChart
from fangbot.chart.parser import ChartParser, get_chart_tool_definition
from fangbot.chart.prompts import EXTRACTION_TOOL_SCHEMA
from fangbot.models import Message, ProviderResponse, Role, ToolCall, ToolDefinition, ToolResult


# -- Mock provider that returns extraction tool_use --


class ExtractionMockProvider(LLMProvider):
    """Mock provider that returns a submit_chart_extraction tool call."""

    def __init__(self, extraction_data: dict):
        self._extraction_data = extraction_data

    @property
    def model_name(self) -> str:
        return "extraction-mock"

    async def call(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        system: str | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc_extract_1",
                    name="submit_chart_extraction",
                    arguments=self._extraction_data,
                )
            ],
            stop_reason="tool_use",
            model="extraction-mock",
        )

    def format_tool_result(self, result: ToolResult) -> Message:
        return Message(role=Role.USER, content=result.content, tool_call_id=result.tool_call_id)


class TestChartParser:
    @pytest.mark.asyncio
    async def test_parse_extracts_facts(self):
        extraction = {
            "facts": [
                {
                    "name": "Creatinine",
                    "value": "1.8 mg/dL",
                    "category": "lab",
                    "timestamp": "2026-03-07T08:14:00Z",
                    "source": "BMP from 2026-03-07",
                    "source_location": "Lab results section",
                    "status": "active",
                    "confidence": 0.95,
                },
                {
                    "name": "Hypertension",
                    "value": "documented",
                    "category": "diagnosis",
                    "timestamp": None,
                    "source": "Past medical history",
                    "source_location": "PMH section",
                    "status": "active",
                    "confidence": 0.9,
                },
            ],
            "warnings": [],
        }
        provider = ExtractionMockProvider(extraction)
        parser = ChartParser(provider)
        chart = await parser.parse("Patient has Cr 1.8, history of HTN")

        assert isinstance(chart, PatientChart)
        assert len(chart.facts) == 2
        assert chart.facts[0].name == "Creatinine"
        assert chart.facts[0].category == FactCategory.LAB
        assert chart.facts[0].confidence == 0.95
        assert chart.facts[1].name == "Hypertension"
        assert chart.facts[1].status == FactStatus.ACTIVE
        assert chart.raw_text == "Patient has Cr 1.8, history of HTN"

    @pytest.mark.asyncio
    async def test_parse_includes_warnings(self):
        extraction = {
            "facts": [
                {
                    "name": "Creatinine",
                    "value": "1.2 mg/dL",
                    "category": "lab",
                    "source": "BMP 3/5",
                },
            ],
            "warnings": ["Creatinine found in two places with different values (1.2 and 1.8)"],
        }
        provider = ExtractionMockProvider(extraction)
        parser = ChartParser(provider)
        chart = await parser.parse("Cr 1.2 ... Cr 1.8")

        assert len(chart.parse_warnings) == 1
        assert "different values" in chart.parse_warnings[0]

    @pytest.mark.asyncio
    async def test_parse_handles_empty_extraction(self):
        extraction = {"facts": [], "warnings": ["No clinical facts found in text"]}
        provider = ExtractionMockProvider(extraction)
        parser = ChartParser(provider)
        chart = await parser.parse("Hello, how are you?")

        assert len(chart.facts) == 0
        assert len(chart.parse_warnings) == 1

    @pytest.mark.asyncio
    async def test_parse_handles_missing_optional_fields(self):
        extraction = {
            "facts": [
                {
                    "name": "HR",
                    "value": "88 bpm",
                    "category": "vital",
                    "source": "Vitals",
                },
            ],
            "warnings": [],
        }
        provider = ExtractionMockProvider(extraction)
        parser = ChartParser(provider)
        chart = await parser.parse("HR 88")

        assert chart.facts[0].timestamp is None
        assert chart.facts[0].source_location is None
        assert chart.facts[0].status is None
        assert chart.facts[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_parse_skips_invalid_facts_gracefully(self):
        extraction = {
            "facts": [
                {
                    "name": "HR",
                    "value": "88 bpm",
                    "category": "vital",
                    "source": "Vitals",
                },
                {
                    "name": "Bad",
                    "value": "data",
                    "category": "not_a_real_category",
                    "source": "???",
                },
            ],
            "warnings": [],
        }
        provider = ExtractionMockProvider(extraction)
        parser = ChartParser(provider)
        chart = await parser.parse("HR 88, some bad data")

        assert len(chart.facts) == 1
        assert chart.facts[0].name == "HR"
        assert any("skipped" in w.lower() or "invalid" in w.lower() for w in chart.parse_warnings)

    @pytest.mark.asyncio
    async def test_parse_handles_provider_error(self):
        class FailingProvider(LLMProvider):
            @property
            def model_name(self) -> str:
                return "failing-mock"

            async def call(self, messages, tools=None, system=None):
                raise RuntimeError("API error")

            def format_tool_result(self, result):
                return Message(role=Role.USER, content=result.content)

        parser = ChartParser(FailingProvider())
        chart = await parser.parse("Some clinical text")

        assert len(chart.facts) == 0
        assert any("error" in w.lower() for w in chart.parse_warnings)

    @pytest.mark.asyncio
    async def test_parse_handles_no_tool_calls_in_response(self):
        class TextOnlyProvider(LLMProvider):
            @property
            def model_name(self) -> str:
                return "text-only-mock"

            async def call(self, messages, tools=None, system=None):
                return ProviderResponse(
                    content="I found some facts but didn't call the tool.",
                    tool_calls=[],
                    stop_reason="end_turn",
                )

            def format_tool_result(self, result):
                return Message(role=Role.USER, content=result.content)

        parser = ChartParser(TextOnlyProvider())
        chart = await parser.parse("Clinical text")

        assert len(chart.facts) == 0
        assert any("extraction" in w.lower() for w in chart.parse_warnings)


class TestExtractionToolSchema:
    def test_schema_has_required_properties(self):
        props = EXTRACTION_TOOL_SCHEMA["input_schema"]["properties"]
        assert "facts" in props
        assert "warnings" in props
        required = EXTRACTION_TOOL_SCHEMA["input_schema"]["required"]
        assert "facts" in required
        assert "warnings" in required

    def test_fact_schema_has_all_fields(self):
        fact_props = EXTRACTION_TOOL_SCHEMA["input_schema"]["properties"]["facts"]["items"][
            "properties"
        ]
        expected_fields = {
            "name",
            "value",
            "category",
            "timestamp",
            "source",
            "source_location",
            "status",
            "confidence",
        }
        assert expected_fields == set(fact_props.keys())


class TestGetChartToolDefinition:
    def test_get_tool_definition(self):
        tool_def = get_chart_tool_definition()
        assert tool_def.name == "parse_patient_chart"
        assert "clinical_text" in tool_def.input_schema["properties"]
        assert "clinical_text" in tool_def.input_schema["required"]
