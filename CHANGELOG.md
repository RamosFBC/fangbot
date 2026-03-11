# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-11

### Added

- Claude Code-style live progress rendering in chat — shows thinking steps, tool calls with arguments, tool results with summaries, and guardrail corrections in real time
- Progress callback protocol (`ProgressCallback`) for decoupled ReAct loop event reporting
- Rich terminal renderer with Markdown synthesis, timing, and compact tool output previews
- Clinical skill system — 4 ambulatory encounter skills (initial consultation, follow-up, medication review, risk assessment)
- Clinical skill loader with internal tool routing in ReAct loop
- System prompt with clinical skill awareness
- Encounter-level evaluation framework — gold standard cases, result models, and decision tracking
- 7 encounter-level evaluation metrics
- 5 initial consultation gold standard encounter cases

### Changed

- Suppress noisy third-party loggers (httpx, openai, anthropic, mcp) using `force=True` on `basicConfig`
- Bump `open-medicine` dependency to `>=0.9.0`
- Reorganize tests into unit/calculator/encounter directories and studies into calculator/encounter

### Fixed

- `logging.basicConfig()` no-op when SDK libraries pre-configure root logger — now uses `force=True` to guarantee configuration

## [0.2.1] - 2026-03-10

### Fixed

- Make `openai` a required dependency instead of optional — fixes `ModuleNotFoundError` when using the OpenAI or local providers
- Remove redundant `openai` and `local` optional dependency groups

## [0.2.0] - 2026-03-10

### Added

- Batch evaluation framework for running gold standard cases through the ReAct agent
- Gold standard case format — YAML-based clinical cases with expected scores, risk tiers, tool calls, and variables
- Evaluation metrics engine — accuracy, MAE, Cohen's Kappa, sensitivity/specificity per risk tier, protocol adherence, CoT quality
- Markdown report generator for cross-provider comparison with summary tables and per-case detail
- `fangbot run` CLI command — runs a study config against gold standard cases
- `fangbot report` CLI command — generates comparison reports from saved results
- CHA2DS2-VASc study with 5 synthetic gold standard cases (scores 0–7, low/moderate/high risk tiers)
- Study configuration format with YAML-based config files
- 61 new tests for the evaluation framework (123 total)

## [0.1.0] - 2026-03-09

### Added

- Initial release of Fangbot
- ReAct loop engine for clinical reasoning with mandatory MCP tool use
- Anthropic Claude provider (Opus 4.6, Sonnet 4.6, Haiku 4.5, Sonnet 4, Opus 4)
- OpenAI provider (GPT-5, GPT-4o, GPT-4.1, o3, o4-mini)
- MCP client connecting to OpenMedicine server via stdio
- Interactive CLI with `fangbot chat` command
- Slash commands: `/help`, `/status`, `/claude`, `/openai`, `/model`, `/models`, `/clear`, `/history`, `/compact`
- Interactive model picker with categorized display
- `fangbot init` setup wizard — provider config, API key, MCP connection test
- `~/.fangbot/` config home for settings and logs
- JSONL audit trail for all agent interactions
- Clinical guardrails enforcing mandatory tool use
- Session context with conversation history management
- 62 unit tests with full mock coverage

[0.3.0]: https://github.com/RamosFBC/fangbot/releases/tag/v0.3.0
[0.2.1]: https://github.com/RamosFBC/fangbot/releases/tag/v0.2.1
[0.2.0]: https://github.com/RamosFBC/fangbot/releases/tag/v0.2.0
[0.1.0]: https://github.com/RamosFBC/fangbot/releases/tag/v0.1.0
