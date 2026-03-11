"""Progress callback protocol for the ReAct loop.

Allows the gateway layer to observe and render agent steps in real time
without coupling the brain to any specific UI framework.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Observer that receives events as the ReAct loop executes."""

    def on_thinking(self, text: str) -> None:
        """Agent produced reasoning text."""
        ...

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Agent is about to call a tool."""
        ...

    def on_tool_result(self, name: str, output: str, is_error: bool = False) -> None:
        """Tool call completed."""
        ...

    def on_iteration(self, iteration: int, max_iterations: int) -> None:
        """A new ReAct iteration is starting."""
        ...

    def on_guardrail_correction(self, violations: list[str]) -> None:
        """Guardrails failed; attempting corrective pass."""
        ...


class NullProgress:
    """No-op callback — used when no renderer is attached."""

    def on_thinking(self, text: str) -> None:
        pass

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    def on_tool_result(self, name: str, output: str, is_error: bool = False) -> None:
        pass

    def on_iteration(self, iteration: int, max_iterations: int) -> None:
        pass

    def on_guardrail_correction(self, violations: list[str]) -> None:
        pass
