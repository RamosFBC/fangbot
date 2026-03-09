"""CLI gateway — interactive chat with slash commands and batch evaluation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fangbot.config import Settings, get_settings
from fangbot.gateway.models_catalog import (
    CATEGORY_STYLES,
    LOCAL_PRESETS,
    PROVIDER_DEFAULTS,
    PROVIDER_MODELS,
    ModelInfo,
)

app = typer.Typer(name="fangbot", help="Fangbot — clinical reasoning powered by OpenMedicine")
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _create_provider(name: str, settings: Settings, model: str | None = None):
    """Create an LLM provider by name."""
    resolved_model = model or PROVIDER_DEFAULTS.get(name, "")

    if name == "claude":
        from fangbot.brain.providers.claude import ClaudeProvider

        return ClaudeProvider(api_key=settings.anthropic_api_key or None, model=resolved_model)
    elif name == "openai":
        from fangbot.brain.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=settings.openai_api_key or None, model=resolved_model)
    elif name in LOCAL_PRESETS:
        from fangbot.brain.providers.local import LocalProvider

        default_url, default_model = LOCAL_PRESETS[name]
        base_url = settings.local_base_url or default_url
        api_key = settings.local_api_key or "not-needed"
        resolved_model = model or resolved_model or default_model
        return LocalProvider(base_url=base_url, model=resolved_model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {name}")


# ---------------------------------------------------------------------------
# Chat state — mutable context that slash commands can modify
# ---------------------------------------------------------------------------


@dataclass
class ChatState:
    settings: Settings
    provider: object  # LLMProvider
    provider_name: str
    react: object  # ReActLoop
    session: object  # SessionContext
    audit: object  # AuditLogger
    tools: list
    mcp: object  # OpenMedicineMCPClient

    def _rebuild_react(self) -> None:
        from fangbot.brain.react import ReActLoop

        self.react = ReActLoop(
            provider=self.provider,
            mcp_client=self.mcp,
            audit_logger=self.audit,
            max_iterations=self.settings.max_iterations,
        )


# ---------------------------------------------------------------------------
# Interactive model selector
# ---------------------------------------------------------------------------


def _select_model_interactive(provider_name: str, current_model: str) -> str | None:
    """Show a numbered list of models and let the user pick one. Returns model ID or None."""
    models = PROVIDER_MODELS.get(provider_name)
    if not models:
        console.print(f"[red]No model catalog for provider '{provider_name}'[/red]")
        return None

    # Group by category
    categories: dict[str, list[tuple[int, ModelInfo]]] = {}
    for idx, m in enumerate(models, 1):
        categories.setdefault(m.category, []).append((idx, m))

    # Display table
    table = Table(
        title=f"Available Models — {provider_name}",
        border_style="dim",
        show_header=True,
        padding=(0, 1),
    )
    table.add_column("#", style="bold", width=4, justify="right")
    table.add_column("Model ID", min_width=28)
    table.add_column("Description")
    table.add_column("Type", width=10)

    for category_name, entries in categories.items():
        style = CATEGORY_STYLES.get(category_name, "")
        for idx, m in entries:
            marker = " *" if m.id == current_model else ""
            id_text = Text(m.id + marker, style=style)
            table.add_row(str(idx), id_text, m.description, category_name)

    console.print()
    console.print(table)
    console.print("[dim]  * = current model[/dim]")
    console.print()

    # Prompt for selection
    try:
        choice = console.input("[bold cyan]Select model number (or Enter to cancel):[/bold cyan] ")
    except (EOFError, KeyboardInterrupt):
        return None

    choice = choice.strip()
    if not choice:
        return None

    # Accept number or model ID directly
    try:
        num = int(choice)
        if 1 <= num <= len(models):
            return models[num - 1].id
        console.print(f"[red]Invalid number. Choose 1-{len(models)}.[/red]")
        return None
    except ValueError:
        # Check if they typed a model ID directly
        valid_ids = {m.id for m in models}
        if choice in valid_ids:
            return choice
        console.print(f"[red]Unknown model: {choice}[/red]")
        return None


# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

CommandHandler = Callable[[ChatState, list[str]], Awaitable[bool]]
_commands: dict[str, tuple[CommandHandler, str]] = {}


def slash_command(name: str, description: str):
    """Decorator to register a slash command."""

    def decorator(fn: CommandHandler) -> CommandHandler:
        _commands[name] = (fn, description)
        return fn

    return decorator


@slash_command("help", "Show available commands")
async def _cmd_help(state: ChatState, args: list[str]) -> bool:
    table = Table(title="Slash Commands", border_style="dim", show_header=True)
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    for cmd_name, (_, desc) in sorted(_commands.items()):
        table.add_row(f"/{cmd_name}", desc)
    table.add_row("quit / exit / q", "End the session")
    console.print(table)
    return True


@slash_command("status", "Show current provider, model, and session info")
async def _cmd_status(state: ChatState, args: list[str]) -> bool:
    console.print(
        Panel(
            f"[bold]Provider:[/bold]  {state.provider_name}\n"
            f"[bold]Model:[/bold]     {state.provider.model_name}\n"
            f"[bold]Session:[/bold]   {state.audit.session_id}\n"
            f"[bold]Tools:[/bold]     {len(state.tools)} discovered\n"
            f"[bold]History:[/bold]   {len(state.session.messages)} messages\n"
            f"[bold]Audit log:[/bold] {state.audit.file_path}",
            title="Status",
            border_style="blue",
        )
    )
    return True


@slash_command("claude", "Switch to Claude (optional: /claude <model>)")
async def _cmd_claude(state: ChatState, args: list[str]) -> bool:
    if args:
        model = args[0]
    else:
        model = _select_model_interactive("claude", state.provider.model_name)
        if model is None:
            console.print("[dim]Cancelled.[/dim]")
            return True
    try:
        state.provider = _create_provider("claude", state.settings, model)
        state.provider_name = "claude"
        state._rebuild_react()
        console.print(f"[green]Switched to Claude ({state.provider.model_name})[/green]")
    except Exception as e:
        console.print(f"[red]Failed to switch: {e}[/red]")
    return True


@slash_command("openai", "Switch to OpenAI (optional: /openai <model>)")
async def _cmd_openai(state: ChatState, args: list[str]) -> bool:
    if args:
        model = args[0]
    else:
        model = _select_model_interactive("openai", state.provider.model_name)
        if model is None:
            console.print("[dim]Cancelled.[/dim]")
            return True
    try:
        state.provider = _create_provider("openai", state.settings, model)
        state.provider_name = "openai"
        state._rebuild_react()
        console.print(f"[green]Switched to OpenAI ({state.provider.model_name})[/green]")
    except Exception as e:
        console.print(f"[red]Failed to switch: {e}[/red]")
    return True


async def _switch_local(state: ChatState, args: list[str], preset: str) -> bool:
    """Switch to a local provider preset, with optional model selection."""
    if args:
        model = args[0]
    else:
        # Try dynamic discovery
        from fangbot.gateway.models_catalog import LOCAL_PRESETS, discover_local_models

        default_url, _ = LOCAL_PRESETS[preset]
        base_url = state.settings.local_base_url or default_url
        console.print(f"[dim]Querying {base_url} for available models...[/dim]")

        models = await discover_local_models(base_url)
        if models:
            PROVIDER_MODELS[preset] = models
            model = _select_model_interactive(preset, getattr(state.provider, "model_name", ""))
            if model is None:
                console.print("[dim]Cancelled.[/dim]")
                return True
        else:
            console.print("[yellow]Could not reach server — enter model name manually.[/yellow]")
            try:
                model = console.input("[bold cyan]Model name:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                return True
            if not model:
                console.print("[dim]Cancelled.[/dim]")
                return True

    try:
        state.provider = _create_provider(preset, state.settings, model)
        state.provider_name = preset
        state._rebuild_react()
        console.print(f"[green]Switched to {preset} ({state.provider.model_name})[/green]")
    except Exception as e:
        console.print(f"[red]Failed to switch: {e}[/red]")
    return True


@slash_command("local", "Switch to local LLM (optional: /local <model>)")
async def _cmd_local(state: ChatState, args: list[str]) -> bool:
    return await _switch_local(state, args, "local")


@slash_command("ollama", "Switch to Ollama (optional: /ollama <model>)")
async def _cmd_ollama(state: ChatState, args: list[str]) -> bool:
    return await _switch_local(state, args, "ollama")


@slash_command("lmstudio", "Switch to LM Studio (optional: /lmstudio <model>)")
async def _cmd_lmstudio(state: ChatState, args: list[str]) -> bool:
    return await _switch_local(state, args, "lmstudio")


@slash_command("vllm", "Switch to vLLM (optional: /vllm <model>)")
async def _cmd_vllm(state: ChatState, args: list[str]) -> bool:
    return await _switch_local(state, args, "vllm")


@slash_command("model", "Switch model — interactive picker or /model <name>")
async def _cmd_model(state: ChatState, args: list[str]) -> bool:
    if args:
        # Direct model switch
        try:
            state.provider = _create_provider(state.provider_name, state.settings, args[0])
            state._rebuild_react()
            console.print(f"[green]Model switched to {state.provider.model_name}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to switch model: {e}[/red]")
        return True

    # Interactive selection
    model = _select_model_interactive(state.provider_name, state.provider.model_name)
    if model is None:
        console.print("[dim]Cancelled.[/dim]")
        return True

    try:
        state.provider = _create_provider(state.provider_name, state.settings, model)
        state._rebuild_react()
        console.print(f"[green]Model switched to {state.provider.model_name}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to switch model: {e}[/red]")
    return True


@slash_command("models", "List all available models for current provider")
async def _cmd_models(state: ChatState, args: list[str]) -> bool:
    provider = args[0] if args else state.provider_name
    models = PROVIDER_MODELS.get(provider)
    if not models:
        console.print(
            f"[red]No model catalog for '{provider}'. Available: {', '.join(PROVIDER_MODELS)}[/red]"
        )
        return True

    table = Table(
        title=f"Models — {provider}",
        border_style="dim",
        show_header=True,
        padding=(0, 1),
    )
    table.add_column("Model ID", min_width=28)
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Type", width=10)

    for m in models:
        style = CATEGORY_STYLES.get(m.category, "")
        marker = " [bold green](active)[/bold green]" if m.id == state.provider.model_name else ""
        table.add_row(
            Text(m.id, style=style),
            m.name + marker,
            m.description,
            m.category,
        )

    console.print(table)
    return True


@slash_command("clear", "Clear conversation history (start fresh)")
async def _cmd_clear(state: ChatState, args: list[str]) -> bool:
    state.session.clear()
    console.print("[green]Conversation history cleared.[/green]")
    return True


@slash_command("history", "Show conversation message count and tool calls")
async def _cmd_history(state: ChatState, args: list[str]) -> bool:
    msgs = state.session.messages
    tools = state.session.tool_calls_made
    console.print(f"[dim]Messages: {len(msgs)} | Tool calls: {len(tools)}[/dim]")
    if tools:
        console.print(f"[dim]Tools used: {', '.join(tools)}[/dim]")
    return True


@slash_command("compact", "Summarize and compress conversation history")
async def _cmd_compact(state: ChatState, args: list[str]) -> bool:
    msg_count = len(state.session.messages)
    if msg_count <= 2:
        console.print("[dim]History too short to compact.[/dim]")
        return True
    tool_calls = state.session.tool_calls_made.copy()
    state.session.clear()
    for tc in tool_calls:
        state.session.record_tool_call(tc)
    console.print(f"[green]Compacted {msg_count} messages. Tool call history preserved.[/green]")
    return True


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------


async def _handle_slash_command(user_input: str, state: ChatState) -> bool:
    """Parse and execute a slash command. Returns True if handled."""
    parts = user_input[1:].split()
    cmd_name = parts[0].lower()
    cmd_args = parts[1:]

    if cmd_name in _commands:
        handler, _ = _commands[cmd_name]
        await handler(state, cmd_args)
        return True

    # Fuzzy match
    matches = [c for c in _commands if c.startswith(cmd_name)]
    if len(matches) == 1:
        handler, _ = _commands[matches[0]]
        await handler(state, cmd_args)
        return True

    console.print(f"[red]Unknown command: /{cmd_name}[/red] — type /help for available commands")
    return True


async def _chat_async() -> None:
    """Async implementation of the interactive chat session."""
    from fangbot.brain.react import ReActLoop
    from fangbot.brain.system_prompt import CLINICAL_SYSTEM_PROMPT
    from fangbot.memory.audit import AuditLogger
    from fangbot.memory.session import SessionContext
    from fangbot.skills.mcp_client import OpenMedicineMCPClient
    from fangbot.skills.tool_registry import ToolRegistry

    settings = get_settings()
    _setup_logging(settings.log_level)

    try:
        provider = _create_provider(settings.provider, settings, settings.model)
    except ValueError:
        console.print(f"[red]Unknown provider: {settings.provider}[/red]")
        raise typer.Exit(1)

    audit = AuditLogger(log_dir=settings.log_dir)
    session_id = audit.start_session()
    session = SessionContext(system_prompt=CLINICAL_SYSTEM_PROMPT)

    console.print(
        Panel(
            f"[bold]Fangbot[/bold]\n"
            f"Provider: {provider.model_name} | Session: {session_id}\n"
            f"Type [bold cyan]/help[/bold cyan] for commands, [bold cyan]/model[/bold cyan] to switch models, "
            f"[bold cyan]quit[/bold cyan] to exit.",
            title="Fangbot",
            border_style="blue",
        )
    )

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

        state = ChatState(
            settings=settings,
            provider=provider,
            provider_name=settings.provider,
            react=react,
            session=session,
            audit=audit,
            tools=tools,
            mcp=mcp,
        )

        while True:
            try:
                user_input = console.input("[bold green]>[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            stripped = user_input.strip()
            if not stripped:
                continue

            if stripped.lower() in ("quit", "exit", "q"):
                break

            # Slash commands
            if stripped.startswith("/"):
                await _handle_slash_command(stripped, state)
                continue

            # Regular message — run through ReAct loop
            try:
                with console.status(
                    f"[bold cyan]{state.provider.model_name} thinking...[/bold cyan]"
                ):
                    result = await state.react.run(stripped, state.session, state.tools)
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                console.print(
                    "[dim]The session is still active. Try /model to switch models or /clear to reset.[/dim]\n"
                )
                continue

            if not result.guardrail_passed:
                console.print(
                    Panel(
                        "\n".join(result.guardrail_violations),
                        title="Guardrail Warnings",
                        border_style="yellow",
                    )
                )

            console.print(f"\n{result.synthesis}\n")
            console.print(
                f"[dim]{state.provider.model_name} · "
                f"tools: {', '.join(result.tool_calls_made) or 'none'} · "
                f"iterations: {result.iterations}[/dim]\n"
            )

    console.print(f"\n[dim]Audit log saved to: {audit.file_path}[/dim]")


@app.command()
def init() -> None:
    """Set up Fangbot — configure provider, API keys, and test MCP connection."""
    from fangbot.gateway.setup import run_setup

    run_setup()


@app.command()
def chat() -> None:
    """Start an interactive clinical reasoning session."""
    asyncio.run(_chat_async())


@app.command()
def run(config: Path = typer.Argument(..., help="Path to study config YAML")) -> None:
    """Run a batch evaluation study against gold standard cases."""
    asyncio.run(_run_study(config))


async def _run_study(config_path: Path) -> None:
    """Execute a batch evaluation study."""
    from fangbot.evaluation.batch_runner import BatchRunner
    from fangbot.evaluation.gold_standard import load_cases, load_study_config

    try:
        console.print(f"[bold]Loading study config:[/bold] {config_path}")
        study_config = load_study_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error loading config:[/red] {exc}")
        raise typer.Exit(1)

    try:
        cases_dir = Path(study_config.cases_dir)
        if not cases_dir.is_absolute():
            cases_dir = config_path.parent / cases_dir
        console.print(f"[bold]Loading cases from:[/bold] {cases_dir}")
        cases = load_cases(cases_dir)
        console.print(f"[green]Loaded {len(cases)} cases[/green]")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error loading cases:[/red] {exc}")
        raise typer.Exit(1)

    runner = BatchRunner(study_config)

    for provider_name in study_config.providers:
        model_name = study_config.models.get(provider_name, _default_model(provider_name))
        console.print(f"\n[bold cyan]Running {provider_name}/{model_name}[/bold cyan]")
        console.print(f"  Cases: {len(cases)}")

        try:
            results = await runner.run_cases(cases, provider_name, model_name)
            saved_path = runner.save_results(results, provider_name, model_name)
            console.print(f"  [green]Results saved:[/green] {saved_path}")
        except Exception as exc:
            console.print(f"  [red]Error running {provider_name}:[/red] {exc}")
            continue

    console.print("\n[bold green]Study complete.[/bold green]")


def _default_model(provider_name: str) -> str:
    """Return default model name for a provider."""
    defaults = {
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "local": "default",
    }
    return defaults.get(provider_name, "unknown")


@app.command()
def report(
    results_dir: Path = typer.Argument(..., help="Path to results directory"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to study config YAML (for gold standard cases)"),
) -> None:
    """Generate a comparison report from evaluation results."""
    import json

    from fangbot.evaluation.gold_standard import load_cases, load_study_config
    from fangbot.evaluation.models import CaseResult
    from fangbot.evaluation.report import generate_report

    if not results_dir.is_dir():
        console.print(f"[red]Error: Results directory not found:[/red] {results_dir}")
        raise typer.Exit(1)

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        console.print(f"[red]No result JSON files found in {results_dir}[/red]")
        raise typer.Exit(1)

    golds = None
    study_name = "Evaluation"
    if config:
        try:
            study_config = load_study_config(config)
            study_name = study_config.study_name
            cases_dir = Path(study_config.cases_dir)
            if not cases_dir.is_absolute():
                cases_dir = config.parent / cases_dir
            golds = load_cases(cases_dir)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[yellow]Warning: Could not load gold standard cases: {exc}[/yellow]")

    results_by_provider: dict[str, list[CaseResult]] = {}
    for jf in json_files:
        data = json.loads(jf.read_text())
        provider_key = f"{data['provider']}/{data['model']}"
        case_results = [CaseResult(**c) for c in data["cases"]]
        results_by_provider[provider_key] = case_results
        if not study_name or study_name == "Evaluation":
            study_name = data.get("study_name", "Evaluation")

    if golds:
        report_text = generate_report(study_name, golds, results_by_provider)
        report_path = results_dir / "report.md"
        report_path.write_text(report_text)
        console.print(f"[green]Report saved to:[/green] {report_path}")
        console.print(report_text)
    else:
        console.print("[yellow]No gold standard config provided — showing raw results summary.[/yellow]")
        for provider_key, results in results_by_provider.items():
            console.print(f"\n[bold]{provider_key}[/bold]: {len(results)} cases")
            errors = sum(1 for r in results if r.error)
            console.print(f"  Errors: {errors}")


if __name__ == "__main__":
    app()
