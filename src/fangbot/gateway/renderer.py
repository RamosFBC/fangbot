"""Rich terminal renderer for ReAct loop progress — Claude Code inspired UX."""

from __future__ import annotations

import json
import textwrap
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Max characters of tool output to show inline
TOOL_OUTPUT_PREVIEW_LINES = 8
TOOL_OUTPUT_MAX_CHARS = 600
THINKING_MAX_CHARS = 300


def _format_args_compact(arguments: dict[str, Any]) -> str:
    """Format tool arguments as a compact, readable string."""
    if not arguments:
        return ""
    parts = []
    for k, v in arguments.items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:57] + "..."
        parts.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
    return ", ".join(parts)


def _truncate_output(text: str) -> tuple[str, bool]:
    """Truncate tool output for display. Returns (display_text, was_truncated)."""
    lines = text.splitlines()
    if len(lines) <= TOOL_OUTPUT_PREVIEW_LINES and len(text) <= TOOL_OUTPUT_MAX_CHARS:
        return text.strip(), False

    # Truncate by lines first, then by chars
    preview_lines = lines[:TOOL_OUTPUT_PREVIEW_LINES]
    preview = "\n".join(preview_lines)
    if len(preview) > TOOL_OUTPUT_MAX_CHARS:
        preview = preview[:TOOL_OUTPUT_MAX_CHARS]

    return preview.strip(), True


class ChatRenderer:
    """Renders ReAct loop events to the terminal in real time.

    Implements the ProgressCallback protocol — pass an instance
    to ReActLoop.run(progress=renderer).
    """

    def __init__(self, console: Console, model_name: str = ""):
        self._console = console
        self._model_name = model_name
        self._iteration = 0
        self._tool_count = 0
        self._start_time: float | None = None
        self._has_printed_thinking = False

    def start(self) -> None:
        """Call when a new user query begins processing."""
        self._iteration = 0
        self._tool_count = 0
        self._start_time = time.monotonic()
        self._has_printed_thinking = False

    # -- ProgressCallback implementation --

    def on_iteration(self, iteration: int, max_iterations: int) -> None:
        self._iteration = iteration

    def on_thinking(self, text: str) -> None:
        if not text or not text.strip():
            return

        # Show a condensed thinking preview
        clean = text.strip()
        if len(clean) > THINKING_MAX_CHARS:
            display = clean[:THINKING_MAX_CHARS].rsplit(" ", 1)[0] + "..."
        else:
            display = clean

        # Only show thinking label on first thinking block per query
        if not self._has_printed_thinking:
            self._console.print()
            self._has_printed_thinking = True

        styled = Text()
        styled.append("  thinking ", style="dim italic")
        styled.append("│ ", style="dim")
        # Indent continuation lines
        wrapped = textwrap.fill(display, width=80)
        first_line, *rest = wrapped.splitlines()
        styled.append(first_line, style="dim")
        self._console.print(styled)
        for line in rest:
            padding = Text("             │ ", style="dim")
            padding.append(line, style="dim")
            self._console.print(padding)

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self._tool_count += 1
        args_str = _format_args_compact(arguments)

        line = Text()
        line.append("  tool call ", style="bold cyan")
        line.append("│ ", style="dim")
        line.append(name, style="bold")
        if args_str:
            line.append(f"({args_str})", style="dim")

        self._console.print(line)

    def on_tool_result(self, name: str, output: str, is_error: bool = False) -> None:
        preview, truncated = _truncate_output(output)

        if is_error:
            line = Text()
            line.append("  tool error", style="bold red")
            line.append(" │ ", style="dim")
            line.append(name, style="bold red")
            line.append(f": {preview}", style="red")
            self._console.print(line)
            return

        # Show result with indented preview
        header = Text()
        header.append("  tool result", style="green")
        header.append("│ ", style="dim")
        header.append(name, style="dim bold")

        # Try to extract a meaningful one-line summary
        summary = self._extract_summary(preview)
        if summary:
            header.append(f" — {summary}", style="dim")

        self._console.print(header)

        # Show truncated output in a subtle way
        if preview and not summary:
            for out_line in preview.splitlines()[:TOOL_OUTPUT_PREVIEW_LINES]:
                result_line = Text()
                result_line.append("              │ ", style="dim")
                result_line.append(out_line.strip(), style="dim")
                self._console.print(result_line)
            if truncated:
                trunc_line = Text()
                trunc_line.append("              │ ", style="dim")
                trunc_line.append("... (truncated)", style="dim italic")
                self._console.print(trunc_line)

    def on_guardrail_correction(self, violations: list[str]) -> None:
        line = Text()
        line.append("  guardrail ", style="bold yellow")
        line.append("│ ", style="dim")
        line.append("Protocol violation detected — retrying with correction", style="yellow")
        self._console.print(line)
        for v in violations:
            detail = Text()
            detail.append("             │ ", style="dim")
            detail.append(f"• {v}", style="yellow dim")
            self._console.print(detail)

    # -- Display helpers --

    def render_synthesis(self, text: str) -> None:
        """Render the final synthesis response."""
        self._console.print()
        self._console.print(Markdown(text))
        self._console.print()

    def render_footer(
        self,
        model_name: str,
        tool_calls: list[str],
        iterations: int,
    ) -> None:
        """Render the metadata footer after a response."""
        elapsed = ""
        if self._start_time is not None:
            secs = time.monotonic() - self._start_time
            if secs < 1:
                elapsed = f"{secs * 1000:.0f}ms"
            else:
                elapsed = f"{secs:.1f}s"

        parts = [model_name]
        if tool_calls:
            unique_tools = list(dict.fromkeys(tool_calls))  # preserve order, dedupe
            parts.append(f"tools: {', '.join(unique_tools)}")
        parts.append(f"iterations: {iterations}")
        if elapsed:
            parts.append(elapsed)

        self._console.print(f"[dim]  {'  ·  '.join(parts)}[/dim]\n")

    def render_guardrail_warnings(self, violations: list[str]) -> None:
        """Render guardrail violation panel."""
        self._console.print(
            Panel(
                "\n".join(f"• {v}" for v in violations),
                title="⚠ Guardrail Warnings",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    def render_error(self, error: str) -> None:
        """Render an error message."""
        self._console.print(f"\n[bold red]  error[/bold red] [dim]│[/dim] {error}\n")

    def _extract_summary(self, text: str) -> str | None:
        """Try to pull a meaningful one-liner from tool output."""
        # If it looks like JSON, try to get a meaningful field
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
                # Common patterns in OpenMedicine responses
                for key in ("score", "result", "risk_level", "name", "title", "message"):
                    if key in data:
                        val = data[key]
                        if isinstance(val, str) and len(val) < 120:
                            return f"{key}: {val}"
                        elif isinstance(val, (int, float)):
                            return f"{key}: {val}"
                return None
            except (json.JSONDecodeError, TypeError):
                pass

        # If it's short enough, use as-is
        first_line = stripped.splitlines()[0] if stripped else ""
        if len(first_line) < 100:
            return first_line
        return None
