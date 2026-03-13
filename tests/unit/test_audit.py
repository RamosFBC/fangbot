"""Tests for the audit logger and session context."""

from __future__ import annotations

import json


from fangbot.memory.audit import AuditEvent, AuditLogger, EventType
from fangbot.memory.session import SessionContext


class TestAuditLogger:
    def test_start_session_creates_file(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        session_id = logger.start_session()

        assert session_id
        assert logger.file_path is not None
        assert logger.file_path.exists()

    def test_log_appends_jsonl(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.start_session()

        logger.log_think("Analyzing the case")
        logger.log_tool_call("search_clinical_calculators", {"query": "CHA2DS2-VASc"})
        logger.log_tool_result("search_clinical_calculators", "Found calculator")
        logger.log_synthesis("Score is 3")

        lines = logger.file_path.read_text().strip().split("\n")
        # session_start + 4 events
        assert len(lines) == 5

        # Each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "event_type" in data
            assert "session_id" in data

    def test_log_tool_error(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.start_session()

        event = logger.log_tool_error("bad_tool", "not found")
        assert event.event_type == EventType.TOOL_ERROR
        assert event.data["error"] == "not found"

    def test_get_events_reads_back(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.start_session()

        logger.log_think("thought 1")
        logger.log_think("thought 2")

        events = logger.get_events()
        assert len(events) == 3  # session_start + 2 thinks
        assert all(isinstance(e, AuditEvent) for e in events)

    def test_get_events_empty_when_no_session(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        assert logger.get_events() == []


class TestSessionContext:
    def test_add_messages(self):
        session = SessionContext(system_prompt="test prompt")

        session.add_user_message("hello")
        session.add_assistant_message("hi there")

        assert len(session.messages) == 2
        assert session.messages[0].role.value == "user"
        assert session.messages[1].role.value == "assistant"

    def test_system_prompt_preserved(self):
        session = SessionContext(system_prompt="You are a clinical assistant.")
        assert session.system_prompt == "You are a clinical assistant."

    def test_tool_call_tracking(self):
        session = SessionContext()

        session.record_tool_call("search_clinical_calculators")
        session.record_tool_call("execute_clinical_calculator")

        assert session.tool_calls_made == [
            "search_clinical_calculators",
            "execute_clinical_calculator",
        ]

    def test_add_tool_result(self):
        session = SessionContext()
        session.add_tool_result("tc_123", "Score: 3")

        assert len(session.messages) == 1
        assert session.messages[0].tool_call_id == "tc_123"

    def test_clear(self):
        session = SessionContext()
        session.add_user_message("test")
        session.record_tool_call("tool1")
        session.clear()

        assert session.messages == []
        assert session.tool_calls_made == []


class TestConfidenceAuditEvents:
    def test_confidence_assessment_event_type_exists(self):
        assert EventType.CONFIDENCE_ASSESSMENT == "confidence_assessment"

    def test_missing_data_detected_event_type_exists(self):
        assert EventType.MISSING_DATA_DETECTED == "missing_data_detected"

    def test_contradiction_detected_event_type_exists(self):
        assert EventType.CONTRADICTION_DETECTED == "contradiction_detected"

    def test_log_confidence_assessment(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.start_session()

        event = logger.log_confidence_assessment(
            confidence="moderate",
            reasoning="Age was estimated",
            missing_data=["Exact DOB"],
            contradictions=[],
            escalation_recommended=False,
        )

        assert event.event_type == EventType.CONFIDENCE_ASSESSMENT
        assert event.data["confidence"] == "moderate"
        assert event.data["reasoning"] == "Age was estimated"
        assert event.data["missing_data"] == ["Exact DOB"]
        assert event.data["contradictions"] == []
        assert event.data["escalation_recommended"] is False
