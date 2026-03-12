"""Tests for the clinical system prompt."""

from __future__ import annotations

from fangbot.brain.system_prompt import CLINICAL_SYSTEM_PROMPT, build_system_prompt


class TestSystemPrompt:
    def test_base_prompt_contains_mandatory_rules(self) -> None:
        assert "NEVER compute clinical scores" in CLINICAL_SYSTEM_PROMPT
        assert "MCP tools" in CLINICAL_SYSTEM_PROMPT

    def test_build_system_prompt_includes_skill_list(self) -> None:
        skills = [
            {"name": "initial_consultation", "description": "First ambulatory visit"},
            {"name": "follow_up", "description": "Return visit"},
        ]
        prompt = build_system_prompt(available_skills=skills)
        assert "initial_consultation" in prompt
        assert "First ambulatory visit" in prompt
        assert "load_clinical_skill" in prompt

    def test_build_system_prompt_without_skills(self) -> None:
        prompt = build_system_prompt(available_skills=[])
        assert "NEVER compute clinical scores" in prompt
        # Should not have skill section if no skills
        assert "load_clinical_skill" not in prompt

    def test_chart_awareness_section_included(self) -> None:
        prompt = build_system_prompt(chart_parsing_available=True)
        assert "parse_patient_chart" in prompt
        assert "chart grounding" in prompt.lower()

    def test_chart_awareness_section_excluded_by_default(self) -> None:
        prompt = build_system_prompt()
        assert "parse_patient_chart" not in prompt

    def test_chart_and_skills_both_included(self) -> None:
        prompt = build_system_prompt(
            available_skills=[{"name": "test_skill", "description": "A test skill"}],
            chart_parsing_available=True,
        )
        assert "test_skill" in prompt
        assert "parse_patient_chart" in prompt
