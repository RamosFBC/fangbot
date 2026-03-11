"""Tests for the clinical skill loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from fangbot.skills.clinical_loader import ClinicalSkillLoader, SkillNotFoundError


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory with registry and one skill."""
    skill_dir = tmp_path / "clinical"
    skill_dir.mkdir()

    registry = skill_dir / "registry.yaml"
    registry.write_text(
        "skills:\n"
        "  - name: initial_consultation\n"
        '    description: "First ambulatory visit"\n'
        "    encounter_types:\n"
        '      - "new_patient"\n'
        '      - "first_visit"\n'
    )

    skill_file = skill_dir / "initial_consultation.md"
    skill_file.write_text("# Initial Consultation\n\nFramework content here.\n")

    return skill_dir


class TestClinicalSkillLoader:
    def test_load_registry(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        registry = loader.registry
        assert len(registry) == 1
        assert registry[0].name == "initial_consultation"
        assert "new_patient" in registry[0].encounter_types

    def test_list_skills_returns_names_and_descriptions(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        listing = loader.list_skills()
        assert len(listing) == 1
        assert listing[0]["name"] == "initial_consultation"
        assert listing[0]["description"] == "First ambulatory visit"

    def test_load_skill_returns_content(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        content = loader.load_skill("initial_consultation")
        assert "# Initial Consultation" in content
        assert "Framework content here." in content

    def test_load_unknown_skill_raises(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        with pytest.raises(SkillNotFoundError, match="unknown_skill"):
            loader.load_skill("unknown_skill")

    def test_get_tool_definition(self, skills_dir: Path) -> None:
        loader = ClinicalSkillLoader(skills_dir)
        tool_def = loader.get_tool_definition()
        assert tool_def.name == "load_clinical_skill"
        assert "initial_consultation" in tool_def.description
