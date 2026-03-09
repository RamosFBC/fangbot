# Contributing to Fangbot

Thanks for your interest in contributing! Fangbot is an open-source clinical reasoning agent and we welcome contributions of all kinds.

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Getting started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- An API key for at least one LLM provider (Anthropic or OpenAI)

### Development setup

```bash
# Clone the repo
git clone https://github.com/RamosFBC/fangbot.git
cd fangbot

# Install all dependencies (dev + optional providers)
uv sync --extra dev --extra openai

# Run tests
uv run python -m pytest -v

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .
```

### Project structure

```
src/fangbot/
├── gateway/        # CLI interface (typer + rich)
├── brain/          # ReAct loop, providers, guardrails
│   └── providers/  # LLM provider implementations
├── skills/         # MCP client, tool registry
└── memory/         # Session context, audit logger
tests/              # pytest test suite
```

## Development workflow

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/your-feature`
3. **Make your changes** — write tests for new functionality
4. **Run the full test suite**: `uv run python -m pytest -v`
5. **Run the linter**: `uv run ruff check .`
6. **Commit** with a clear message (see conventions below)
7. **Push** and open a Pull Request

### Branch naming

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation |
| `refactor/` | Code restructuring |
| `test/` | Test additions/fixes |

### Commit messages

Use clear, descriptive commit messages:

```
Add renal dosing workflow with CKD-EPI integration

Implements the multi-step renal dosing calculation chain using
CKD-EPI GFR estimation followed by dose adjustment lookup.
```

- Start with a verb (Add, Fix, Update, Remove, Refactor)
- First line under 72 characters
- Body explains **why**, not just **what**

## Testing

```bash
# Full test suite
uv run python -m pytest -v

# Single test file
uv run python -m pytest tests/test_react_loop.py -v

# Specific test
uv run python -m pytest -k "test_guardrail" -v

# Integration tests (requires open-medicine-mcp)
uv run python -m pytest -m integration -v
```

### Writing tests

- Every new feature needs tests
- Use `pytest` fixtures and `tmp_path` for file operations
- Mock external services (LLM APIs, MCP server) — see `tests/conftest.py`
- Use `@pytest.mark.asyncio` for async tests
- Use `@pytest.mark.integration` for tests requiring a live MCP server

## Code style

- **Linter/formatter:** ruff (configured in `pyproject.toml`)
- **Type hints** on all public functions
- **Pydantic models** for all data structures
- **Line length:** 100 characters
- **Target:** Python 3.10+ (no walrus operator in hot paths, use `from __future__ import annotations`)

Run before committing:

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Adding a new LLM provider

1. Create `src/fangbot/brain/providers/your_provider.py`
2. Implement the `LLMProvider` ABC from `base.py`
3. Add the provider to `_create_provider()` in `gateway/cli.py`
4. Add models to `gateway/models_catalog.py`
5. Add the SDK as an optional dependency in `pyproject.toml`
6. Write tests in `tests/test_providers.py`

## Adding a new clinical workflow

1. Create `src/fangbot/workflows/your_workflow.py`
2. Define gold standard test cases
3. Add to the study configuration
4. Write evaluation tests

## Critical rules

- **Never compute clinical scores from LLM knowledge** — always use MCP tools
- **Never import OpenMedicine directly** — communicate via MCP protocol only
- **Synthetic patient data only** — no real patient data (per Resolucao CNS 510/2016)
- **Always include DOI references** from tool results in responses
- **Never downplay contraindication flags** from MCP tools

## Releasing

Releases follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0) — breaking API changes
- **MINOR** (0.2.0) — new features, backward compatible
- **PATCH** (0.1.1) — bug fixes

Releases are automated via GitHub Actions when a version tag is pushed:

```bash
# Update version in pyproject.toml and src/fangbot/__init__.py
# Update CHANGELOG.md
git tag v0.2.0
git push origin v0.2.0
```

## Questions?

Open an [issue](https://github.com/RamosFBC/fangbot/issues) or start a [discussion](https://github.com/RamosFBC/fangbot/discussions).
