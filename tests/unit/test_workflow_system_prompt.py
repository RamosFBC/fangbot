"""Tests for workflow awareness in the system prompt."""

from fangbot.brain.system_prompt import build_system_prompt


class TestWorkflowSystemPrompt:
    def test_no_workflows_no_section(self):
        prompt = build_system_prompt()
        assert "CLINICAL WORKFLOWS" not in prompt

    def test_workflows_included(self):
        workflows = [
            {"name": "admission_oneliner", "description": "Generate admission one-liner"},
            {"name": "handoff_draft", "description": "Generate IPASS handoff"},
        ]
        prompt = build_system_prompt(available_workflows=workflows)
        assert "CLINICAL WORKFLOWS" in prompt
        assert "admission_oneliner" in prompt
        assert "handoff_draft" in prompt
        assert "run_workflow" in prompt

    def test_all_sections_together(self):
        skills = [{"name": "initial_consultation", "description": "First visit"}]
        workflows = [{"name": "pre_round_summary", "description": "Pre-round summary"}]
        prompt = build_system_prompt(
            available_skills=skills,
            chart_parsing_available=True,
            uncertainty_calibration=True,
            available_workflows=workflows,
        )
        assert "CLINICAL REASONING SKILLS" in prompt
        assert "CHART GROUNDING" in prompt
        assert "UNCERTAINTY CALIBRATION" in prompt
        assert "CLINICAL WORKFLOWS" in prompt
