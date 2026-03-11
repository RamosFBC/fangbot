"""Loader for encounter-based clinical reasoning skills."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from fangbot.models import ToolDefinition

logger = logging.getLogger(__name__)

# Default path — resolved relative to this file
_DEFAULT_SKILLS_DIR = Path(__file__).parent / "clinical"


class SkillNotFoundError(Exception):
    """Raised when a requested clinical skill does not exist."""


class SkillEntry(BaseModel):
    """A single entry in the skill registry."""

    name: str
    description: str
    encounter_types: list[str] = Field(default_factory=list)


class ClinicalSkillLoader:
    """Discovers and loads clinical reasoning skill files."""

    def __init__(self, skills_dir: Path | None = None):
        self._skills_dir = skills_dir or _DEFAULT_SKILLS_DIR
        self._registry: list[SkillEntry] | None = None

    @property
    def registry(self) -> list[SkillEntry]:
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self) -> list[SkillEntry]:
        registry_path = self._skills_dir / "registry.yaml"
        if not registry_path.exists():
            logger.warning(f"No skill registry found at {registry_path}")
            return []
        data = yaml.safe_load(registry_path.read_text())
        return [SkillEntry(**entry) for entry in data.get("skills", [])]

    def list_skills(self) -> list[dict[str, str]]:
        """Return a list of available skills with name and description."""
        return [{"name": entry.name, "description": entry.description} for entry in self.registry]

    def load_skill(self, skill_name: str) -> str:
        """Load the content of a clinical skill by name."""
        valid_names = {entry.name for entry in self.registry}
        if skill_name not in valid_names:
            raise SkillNotFoundError(
                f"Skill '{skill_name}' not found. Available: {sorted(valid_names)}"
            )
        skill_path = self._skills_dir / f"{skill_name}.md"
        if not skill_path.exists():
            raise SkillNotFoundError(f"Skill file not found: {skill_path}")
        return skill_path.read_text()

    def get_tool_definition(self) -> ToolDefinition:
        """Return the ToolDefinition for load_clinical_skill."""
        skill_names = [entry.name for entry in self.registry]
        return ToolDefinition(
            name="load_clinical_skill",
            description=(
                "Load a clinical reasoning framework for the current encounter. "
                f"Available skills: {', '.join(skill_names)}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": skill_names,
                        "description": "The encounter type skill to load",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this skill is relevant to the current encounter",
                    },
                },
                "required": ["skill_name"],
            },
        )
