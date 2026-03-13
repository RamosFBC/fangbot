"""Extraction prompt and tool schema for LLM-assisted chart parsing."""

from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """You are a clinical data extraction engine. Your ONLY job is to parse clinical text and extract ALL discrete clinical facts into structured data.

## Instructions

For each clinical fact found in the text, extract:
- **name**: The clinical entity (e.g., "Creatinine", "Metformin", "Hypertension", "Heart Rate")
- **value**: The value or description (e.g., "1.8 mg/dL", "500mg twice daily", "documented")
- **category**: One of: lab, vital, medication, diagnosis, procedure, allergy, imaging, culture
- **timestamp**: ISO 8601 datetime if identifiable from context, null if not
- **source**: Brief description of where in the text this was found (e.g., "BMP from 2026-03-07", "Medication list")
- **source_location**: Which section of the note (e.g., "Lab results section", "Past medical history", "Assessment")
- **status**: One of: active, historical, resolved — null if unclear
- **confidence**: 0.0 to 1.0 — how confident you are in this extraction

## Rules

1. Extract EVERY discrete clinical fact. Each lab value, each vital sign, each medication, each diagnosis gets its own entry.
2. Do NOT summarize or aggregate — one fact per entry.
3. Distinguish active from historical conditions.
4. For labs/vitals with units, include units in the value.
5. If the same measurement appears multiple times with different values, extract EACH occurrence separately and add a warning.

## Warnings

Add warnings for:
- Conflicting values (same test, different results at similar times)
- Ambiguous temporal references that could not be resolved
- Unclear active vs. historical status on clinically significant findings
- Potential copy-forward artifacts (identical notes across dates)
- Values that seem clinically implausible (e.g., HR of 500)

Call the `submit_chart_extraction` tool with your results."""

EXTRACTION_TOOL_SCHEMA: dict = {
    "name": "submit_chart_extraction",
    "description": "Submit extracted clinical facts from the chart text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Clinical entity name"},
                        "value": {"type": "string", "description": "Value or description"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "lab",
                                "vital",
                                "medication",
                                "diagnosis",
                                "procedure",
                                "allergy",
                                "imaging",
                                "culture",
                            ],
                        },
                        "timestamp": {
                            "type": ["string", "null"],
                            "description": "ISO 8601 datetime if identifiable",
                        },
                        "source": {
                            "type": "string",
                            "description": "Where in the text this was found",
                        },
                        "source_location": {
                            "type": ["string", "null"],
                            "description": "Section of the note",
                        },
                        "status": {
                            "type": ["string", "null"],
                            "enum": ["active", "historical", "resolved", None],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Extraction confidence",
                        },
                    },
                    "required": ["name", "value", "category", "source"],
                },
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extraction warnings (conflicts, ambiguities, implausible values)",
            },
        },
        "required": ["facts", "warnings"],
    },
}
