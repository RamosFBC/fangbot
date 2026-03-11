"""Tests for internal tool routing in the ReAct loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from fangbot.brain.react import ReActLoop
from fangbot.memory.audit import AuditLogger
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


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "clinical"
    skill_dir.mkdir()
    registry = skill_dir / "registry.yaml"
    registry.write_text(
        "skills:\n"
        "  - name: initial_consultation\n"
        '    description: "First ambulatory visit"\n'
        "    encounter_types:\n"
        '      - "new_patient"\n'
    )
    skill_file = skill_dir / "initial_consultation.md"
    skill_file.write_text("# Initial Consultation\n\nDo systematic assessment.\n")
    return skill_dir


class TestToolRouting:
    @pytest.mark.asyncio
    async def test_internal_tool_handled_locally(self, audit_logger, session, skills_dir):
        """load_clinical_skill should be intercepted, not sent to MCP."""
        provider = MockProvider(
            responses=[
                # Agent requests skill load
                ProviderResponse(
                    content="Let me load the clinical framework.",
                    tool_calls=[
                        ToolCall(
                            id="tc_skill",
                            name="load_clinical_skill",
                            arguments={
                                "skill_name": "initial_consultation",
                                "reason": "New patient visit",
                            },
                        )
                    ],
                    stop_reason="tool_use",
                ),
                # Agent then calls MCP tool
                ProviderResponse(
                    content="Now calculating CHA2DS2-VASc.",
                    tool_calls=[
                        ToolCall(
                            id="tc_calc",
                            name="search_clinical_calculators",
                            arguments={"query": "CHA2DS2-VASc"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                # Final synthesis
                ProviderResponse(
                    content="Assessment complete.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()

        from fangbot.skills.clinical_loader import ClinicalSkillLoader

        loader = ClinicalSkillLoader(skills_dir)

        loop = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit_logger,
            clinical_skill_loader=loader,
        )

        from fangbot.models import ToolDefinition

        tools = [
            loader.get_tool_definition(),
            ToolDefinition(
                name="search_clinical_calculators",
                description="Search for clinical calculators",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ]

        result = await loop.run("New patient with AFib", session, tools)

        # Skill load should NOT have been sent to MCP
        mcp_tool_names = [c["name"] for c in mcp.calls]
        assert "load_clinical_skill" not in mcp_tool_names
        # MCP tool should have been called
        assert "search_clinical_calculators" in mcp_tool_names
        # Skill content should be in the conversation
        assert "load_clinical_skill" in result.tool_calls_made

    @pytest.mark.asyncio
    async def test_skill_load_returns_content_to_agent(self, audit_logger, session, skills_dir):
        """The skill content should be returned as a tool result."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Loading skill.",
                    tool_calls=[
                        ToolCall(
                            id="tc_skill",
                            name="load_clinical_skill",
                            arguments={"skill_name": "initial_consultation"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Following the framework now.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()

        from fangbot.skills.clinical_loader import ClinicalSkillLoader

        loader = ClinicalSkillLoader(skills_dir)

        loop = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit_logger,
            clinical_skill_loader=loader,
        )

        tools = [loader.get_tool_definition()]
        await loop.run("Test skill loading", session, tools)

        # Check that the skill content ended up in the session messages
        tool_messages = [m for m in session.messages if m.role.value == "tool"]
        assert any("Initial Consultation" in m.content for m in tool_messages)

    @pytest.mark.asyncio
    async def test_invalid_skill_returns_error(self, audit_logger, session, skills_dir):
        """Loading a non-existent skill should return an error, not crash."""
        provider = MockProvider(
            responses=[
                ProviderResponse(
                    content="Loading skill.",
                    tool_calls=[
                        ToolCall(
                            id="tc_bad",
                            name="load_clinical_skill",
                            arguments={"skill_name": "nonexistent"},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                ProviderResponse(
                    content="Skill not found, proceeding without framework.",
                    stop_reason="end_turn",
                ),
            ]
        )

        mcp = MockMCPClient()

        from fangbot.skills.clinical_loader import ClinicalSkillLoader

        loader = ClinicalSkillLoader(skills_dir)
        loop = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit_logger,
            clinical_skill_loader=loader,
        )

        tools = [loader.get_tool_definition()]
        result = await loop.run("Test bad skill", session, tools)

        # Should complete without crashing
        assert result.synthesis is not None
