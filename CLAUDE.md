# OpenMedicine Agent

An open-source medical reasoning agent that connects to the [OpenMedicine](https://github.com/open-medicine/openmedicine) MCP server to perform deterministic, auditable clinical calculations. **This is NOT a diagnostic tool** — it is a research platform for validating LLM-based clinical tool orchestration.

## Critical Constraint

> The agent MUST call OpenMedicine's deterministic MCP tools for any clinical calculation. It must NEVER compute scores, doses, or risk stratifications from its own parametric knowledge.

Enforced by: system prompt rules, evaluation harness (protocol adherence check), and audit trail verification. Results without tool-call traces are failures regardless of correctness.

## Commands

```bash
uv sync                                        # Install dependencies
uv run agent chat                               # Interactive CLI mode
uv run agent run studies/chadsvasc/config.yaml   # Batch evaluation
uv run agent report studies/chadsvasc/results/   # Generate comparison report
uv run python -m pytest -v                       # Run tests
uv run python -m pytest tests/test_file.py -v    # Run single test file
uv run python -m pytest -k "test_name" -v        # Run specific test
```

## Tech Stack

- **Language:** Python 3.10+
- **Package manager:** uv (matches OpenMedicine)
- **LLM SDKs:** anthropic, openai, google-genai (multi-model benchmarking)
- **MCP client:** mcp (Python SDK) — connects to OpenMedicine via stdio
- **Validation:** Pydantic
- **CLI:** typer or click
- **API:** FastAPI
- **Testing:** pytest + Hypothesis
- **Audit format:** JSON-lines (.jsonl)

## Architecture

```
Gateway → Brain → Skills (MCP) → Memory
                                → Evaluation
```

| Layer | Path | Purpose |
|-------|------|---------|
| Gateway | `src/openmedicine_agent/gateway/` | Case ingestion (CLI, API, file loader) |
| Brain | `src/openmedicine_agent/brain/` | ReAct loop + clinical guardrails + LLM providers |
| Skills | `src/openmedicine_agent/skills/` | MCP client connecting to OpenMedicine server |
| Memory | `src/openmedicine_agent/memory/` | Session context, audit log (JSONL), chain-of-thought |
| Workflows | `src/openmedicine_agent/workflows/` | Study-specific clinical workflows |
| Evaluation | `src/openmedicine_agent/evaluation/` | Batch runner, gold standard comparison, metrics |
| Safety | `src/openmedicine_agent/safety/` | Validators, contraindication flags |

## MCP Integration Rules

- The agent spawns `uv run open-medicine-mcp` as a subprocess and communicates via stdin/stdout
- The agent NEVER imports OpenMedicine directly — always goes through MCP protocol
- Available meta-tools: `search_clinical_calculators`, `execute_clinical_calculator`, `search_guidelines`, `retrieve_guideline`
- OpenMedicine >= 0.4.0 required

## LLM Providers

Abstract base in `brain/providers/base.py`. Implementations:
- `claude.py` — Anthropic Claude (primary)
- `openai.py` — OpenAI GPT-4
- `gemini.py` — Google Gemini
- `ollama.py` — Local models via Ollama

All providers must implement the same interface for benchmark comparability.

## Research Studies

| Study | Workflow | Calculator | Complexity |
|-------|----------|-----------|------------|
| 1: Renal Dosing | `renal_dosing.py` | CKD-EPI + dose adjustment | Multi-step chain |
| 2: Glasgow Coma Score | `gcs_assessment.py` | GCS calculator | Narrative extraction |
| 3: CHA2DS2-VASc | `chadsvasc.py` | CHA2DS2-VASc | Single calculator (simplest) |

Build order: Study 3 first (simplest), then Study 2, then Study 1.

## Evaluation Metrics

- **accuracy** — exact score match
- **mae** — mean absolute error
- **kappa** — Cohen's Kappa (inter-rater reliability)
- **sensitivity/specificity** — per risk tier
- **protocol_adherence** — did the agent call the right tools?
- **cot_quality** — was reasoning auditable and correct?

## Audit Log

Every agent run produces a JSONL audit file with events: `case_received`, `think`, `tool_call`, `tool_result`, `synthesis`, `evaluation`. Each tool result includes DOI references.

## Code Style

- Pydantic models for all data structures
- Type hints on all public functions
- Async where appropriate (MCP client, LLM calls)
- Follow OpenMedicine patterns for consistency

## Gotchas

- MCP communication is stdio-based — never try HTTP/WebSocket to OpenMedicine
- Synthetic patient data only — no real patient data (per Resolucao CNS 510/2016)
- metric_mismatch_warning from tools must be included verbatim in responses
- Contraindication results must be prominently flagged, never downplayed
- Gold standard cases require expert validation before use in evaluation
