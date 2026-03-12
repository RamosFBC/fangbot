"""LLM-assisted chart parser — structured extraction with provenance."""

from __future__ import annotations

import logging

from fangbot.brain.providers.base import LLMProvider
from fangbot.chart.models import ChartFact, PatientChart
from fangbot.chart.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_TOOL_SCHEMA
from fangbot.models import Message, Role, ToolDefinition

logger = logging.getLogger(__name__)


class ChartParser:
    """Extracts structured clinical facts from free text using an LLM provider."""

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    async def parse(self, clinical_text: str) -> PatientChart:
        """Parse clinical text into a structured PatientChart with provenance."""
        extraction_tool = ToolDefinition(**EXTRACTION_TOOL_SCHEMA)
        messages = [Message(role=Role.USER, content=clinical_text)]

        try:
            response = await self._provider.call(
                messages=messages,
                tools=[extraction_tool],
                system=EXTRACTION_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(f"Chart extraction failed: {e}")
            return PatientChart(
                raw_text=clinical_text,
                parse_warnings=[f"Chart extraction error: {e}"],
            )

        # Find the submit_chart_extraction tool call
        extraction_call = next(
            (tc for tc in response.tool_calls if tc.name == "submit_chart_extraction"),
            None,
        )
        if extraction_call is None:
            logger.warning("LLM did not call submit_chart_extraction tool")
            return PatientChart(
                raw_text=clinical_text,
                parse_warnings=[
                    "Chart extraction failed: LLM did not produce structured extraction output"
                ],
            )

        return self._build_chart(clinical_text, extraction_call.arguments)

    def _build_chart(self, raw_text: str, extraction: dict) -> PatientChart:
        """Build a PatientChart from raw extraction data, skipping invalid facts."""
        facts: list[ChartFact] = []
        warnings: list[str] = list(extraction.get("warnings", []))

        for i, raw_fact in enumerate(extraction.get("facts", [])):
            try:
                fact = ChartFact(**raw_fact)
                facts.append(fact)
            except Exception as e:
                msg = f"Skipped invalid fact at index {i}: {e}"
                logger.warning(msg)
                warnings.append(msg)

        return PatientChart(facts=facts, raw_text=raw_text, parse_warnings=warnings)


def get_chart_tool_definition() -> ToolDefinition:
    """Return the ToolDefinition for the parse_patient_chart internal tool."""
    return ToolDefinition(
        name="parse_patient_chart",
        description=(
            "Parse a clinical narrative into structured chart data with provenance tracking. "
            "Returns extracted facts (labs, vitals, medications, diagnoses, procedures, "
            "allergies, imaging, cultures) with source citations. Use this when a patient "
            "case or clinical note is presented and you need to identify discrete clinical facts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "clinical_text": {
                    "type": "string",
                    "description": "The raw clinical text to parse",
                }
            },
            "required": ["clinical_text"],
        },
    )
