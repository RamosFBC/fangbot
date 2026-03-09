"""CLI gateway — interactive chat and batch evaluation commands."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from openmedicine_agent.config import get_settings

app = typer.Typer(name="agent", help="OpenMedicine Agent CLI")
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


async def _chat_async() -> None:
    """Async implementation of the interactive chat session."""
    from openmedicine_agent.brain.providers.claude import ClaudeProvider
    from openmedicine_agent.brain.react import ReActLoop
    from openmedicine_agent.brain.system_prompt import CLINICAL_SYSTEM_PROMPT
    from openmedicine_agent.memory.audit import AuditLogger
    from openmedicine_agent.memory.session import SessionContext
    from openmedicine_agent.skills.mcp_client import OpenMedicineMCPClient
    from openmedicine_agent.skills.tool_registry import ToolRegistry

    settings = get_settings()
    _setup_logging(settings.log_level)

    # Initialize provider
    if settings.provider == "claude":
        provider = ClaudeProvider(api_key=settings.anthropic_api_key or None, model=settings.model)
    else:
        console.print(f"[red]Provider '{settings.provider}' not yet implemented.[/red]")
        raise typer.Exit(1)

    audit = AuditLogger(log_dir=settings.log_dir)
    session_id = audit.start_session()
    session = SessionContext(system_prompt=CLINICAL_SYSTEM_PROMPT)

    console.print(Panel(
        f"[bold]OpenMedicine Agent[/bold]\n"
        f"Provider: {provider.model_name}\n"
        f"Session: {session_id}\n"
        f"Type 'quit' or 'exit' to end.",
        title="Clinical Reasoning Assistant",
        border_style="blue",
    ))

    mcp = OpenMedicineMCPClient(
        command=settings.mcp_command,
        args=settings.mcp_args_list,
    )

    async with mcp.connect():
        registry = ToolRegistry(mcp)
        tools = await registry.get_tools()
        console.print(f"[dim]Discovered {len(tools)} MCP tools: {[t.name for t in tools]}[/dim]\n")

        react = ReActLoop(
            provider=provider,
            mcp_client=mcp,
            audit_logger=audit,
            max_iterations=settings.max_iterations,
        )

        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.strip().lower() in ("quit", "exit", "q"):
                break

            if not user_input.strip():
                continue

            with console.status("[bold cyan]Thinking...[/bold cyan]"):
                result = await react.run(user_input, session, tools)

            # Display result
            if not result.guardrail_passed:
                console.print(Panel(
                    "\n".join(result.guardrail_violations),
                    title="Guardrail Warnings",
                    border_style="yellow",
                ))

            console.print(f"\n[bold blue]Agent:[/bold blue] {result.synthesis}\n")
            console.print(
                f"[dim](Tools called: {result.tool_calls_made or 'none'} | "
                f"Iterations: {result.iterations})[/dim]\n"
            )

    console.print(f"\n[dim]Audit log saved to: {audit.file_path}[/dim]")


@app.command()
def chat() -> None:
    """Start an interactive clinical reasoning session."""
    asyncio.run(_chat_async())


@app.command()
def run(config: Path = typer.Argument(..., help="Path to study config YAML")) -> None:
    """Run a batch evaluation study (Phase 2)."""
    console.print(f"[yellow]Batch evaluation not yet implemented. Config: {config}[/yellow]")
    raise typer.Exit(0)


@app.command()
def report(results_dir: Path = typer.Argument(..., help="Path to results directory")) -> None:
    """Generate a comparison report from evaluation results (Phase 2)."""
    console.print(f"[yellow]Report generation not yet implemented. Dir: {results_dir}[/yellow]")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
