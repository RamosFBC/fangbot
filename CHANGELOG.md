# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/RamosFBC/fangbot/releases/tag/v0.1.0
