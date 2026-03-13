"""JSONL audit logger for recording agent activity."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    SESSION_START = "session_start"
    CASE_RECEIVED = "case_received"
    THINK = "think"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    SKILL_LOADED = "skill_loaded"
    CHART_PARSE = "chart_parse"
    CHART_CONSISTENCY = "chart_consistency"
    SYNTHESIS = "synthesis"


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Append-only JSONL audit logger, one file per session."""

    def __init__(self, log_dir: str | Path = "logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id: str = ""
        self._file_path: Path | None = None

    def start_session(self, session_id: str | None = None) -> str:
        """Initialize a new audit session. Returns the session ID."""
        self._session_id = session_id or uuid4().hex[:16]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._file_path = self._log_dir / f"session_{timestamp}_{self._session_id}.jsonl"
        self.log(EventType.SESSION_START, {"session_id": self._session_id})
        logger.info(f"Audit log: {self._file_path}")
        return self._session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    def log(self, event_type: EventType, data: dict[str, Any] | None = None) -> AuditEvent:
        """Append an event to the JSONL audit file."""
        event = AuditEvent(
            session_id=self._session_id,
            event_type=event_type,
            data=data or {},
        )
        if self._file_path:
            with open(self._file_path, "a") as f:
                f.write(event.model_dump_json() + "\n")
        return event

    def log_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> AuditEvent:
        return self.log(EventType.TOOL_CALL, {"tool_name": tool_name, "arguments": arguments})

    def log_tool_result(self, tool_name: str, result: str) -> AuditEvent:
        return self.log(EventType.TOOL_RESULT, {"tool_name": tool_name, "result": result})

    def log_tool_error(self, tool_name: str, error: str) -> AuditEvent:
        return self.log(EventType.TOOL_ERROR, {"tool_name": tool_name, "error": error})

    def log_think(self, thought: str) -> AuditEvent:
        return self.log(EventType.THINK, {"thought": thought})

    def log_synthesis(self, synthesis: str) -> AuditEvent:
        return self.log(EventType.SYNTHESIS, {"synthesis": synthesis})

    def get_events(self) -> list[AuditEvent]:
        """Read all events from the current session's audit file."""
        if not self._file_path or not self._file_path.exists():
            return []
        events = []
        with open(self._file_path) as f:
            for line in f:
                if line.strip():
                    events.append(AuditEvent.model_validate_json(line))
        return events
