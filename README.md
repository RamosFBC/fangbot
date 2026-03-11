# Fangbot

[![PyPI version](https://img.shields.io/pypi/v/fangbot.svg)](https://pypi.org/project/fangbot/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/RamosFBC/fangbot/actions/workflows/ci.yml/badge.svg)](https://github.com/RamosFBC/fangbot/actions/workflows/ci.yml)

Clinical reasoning agent powered by [OpenMedicine](https://github.com/open-medicine/openmedicine) — fangs of the healing serpent.

**This is NOT a diagnostic tool** — it is a research platform for validating LLM-based clinical tool orchestration.

## What it does

Fangbot connects to the OpenMedicine MCP server to perform deterministic, auditable clinical calculations via a ReAct (Reasoning + Acting) loop. It supports multiple LLM providers for benchmarking.

**Critical constraint:** The agent always calls OpenMedicine's MCP tools for clinical calculations — it never computes scores from its own knowledge.

## Install

```bash
# Recommended: install as a global CLI tool
pipx install fangbot

# Or with uv
uv tool install fangbot

# With OpenAI support
pipx install "fangbot[openai]"
```

## Quick start

```bash
# Interactive setup — configure provider, API key, test MCP connection
fangbot init

# Start a clinical reasoning session
fangbot chat
```

## Usage

```bash
fangbot init                          # First-time setup wizard
fangbot chat                          # Interactive clinical reasoning session
fangbot run studies/chadsvasc/config.yaml   # Batch evaluation against gold standard cases
fangbot report studies/chadsvasc/results/   # Generate cross-provider comparison report
```

### Chat commands

Inside `fangbot chat`, use slash commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/status` | Current provider, model, session info |
| `/claude [model]` | Switch to Claude |
| `/openai [model]` | Switch to OpenAI |
| `/model [name]` | Interactive model picker |
| `/models` | List all available models |
| `/clear` | Clear conversation history |
| `/history` | Show message count and tool calls |
| `/compact` | Compress conversation history |
| `quit` | End the session |

## Configuration

Fangbot stores config in `~/.fangbot/`:

```
~/.fangbot/
├── .env          # API keys and settings
└── logs/         # JSONL audit trail
```

Environment variables (prefix `FANGBOT_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FANGBOT_PROVIDER` | `claude` | LLM provider (`claude`, `openai`) |
| `FANGBOT_MODEL` | `claude-sonnet-4-20250514` | Model to use |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |

## Supported providers

| Provider | Models | Status |
|----------|--------|--------|
| Anthropic Claude | Opus 4.6, Sonnet 4.6, Haiku 4.5, Sonnet 4, Opus 4 | Supported |
| OpenAI | GPT-5, GPT-4o, GPT-4.1, o3, o4-mini | Supported |
| Google Gemini | — | Planned |
| Ollama (local) | — | Planned |

## Architecture

```
Gateway (CLI) → Brain (ReAct loop) → Skills (MCP tools) → Memory (audit trail)
                                                        → Evaluation (batch runner, metrics, reports)
```

- **Gateway** — CLI interface with typer + rich
- **Brain** — ReAct loop engine, LLM providers, clinical guardrails
- **Skills** — MCP client connecting to OpenMedicine server via stdio
- **Memory** — Session context, JSONL audit logger
- **Evaluation** — Batch runner, gold standard comparison, metrics engine, Markdown report generator

## Evaluation

Fangbot includes a batch evaluation framework for benchmarking LLM providers against gold standard clinical cases.

```bash
# Run all CHA2DS2-VASc cases through the agent
fangbot run studies/chadsvasc/config.yaml

# Generate a comparison report from saved results
fangbot report studies/chadsvasc/results/ --config studies/chadsvasc/config.yaml
```

Gold standard cases are YAML files with expected scores, risk tiers, and tool calls. The evaluation engine computes:

| Metric | Description |
|--------|-------------|
| Accuracy | Exact score match rate |
| MAE | Mean absolute error |
| Cohen's Kappa | Inter-rater reliability |
| Sensitivity/Specificity | Per risk tier (low/moderate/high) |
| Protocol adherence | Did the agent call the required MCP tools? |
| CoT quality | Was reasoning auditable with chain-of-thought? |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

```bash
git clone https://github.com/RamosFBC/fangbot.git
cd fangbot
uv sync --extra dev --extra openai
uv run python -m pytest -v
```

## License

[MIT](LICENSE) — Copyright (c) 2026 RamosFBC
