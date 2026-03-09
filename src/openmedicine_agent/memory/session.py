"""Session context for tracking conversation state."""

from __future__ import annotations

from openmedicine_agent.models import Message, Role


class SessionContext:
    """Tracks the conversation messages and tool calls for the current session."""

    def __init__(self, system_prompt: str = ""):
        self._system_prompt = system_prompt
        self._messages: list[Message] = []
        self._tool_calls_made: list[str] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def tool_calls_made(self) -> list[str]:
        return list(self._tool_calls_made)

    def add_user_message(self, content: str) -> None:
        self._messages.append(Message(role=Role.USER, content=content))

    def add_assistant_message(self, content: str, tool_calls=None) -> None:
        self._messages.append(Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls))

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id))

    def record_tool_call(self, tool_name: str) -> None:
        self._tool_calls_made.append(tool_name)

    def clear(self) -> None:
        self._messages.clear()
        self._tool_calls_made.clear()
