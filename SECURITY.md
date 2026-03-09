# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue**
2. Email: [open an issue with the security label](https://github.com/RamosFBC/fangbot/issues) marked as confidential, or contact the maintainer directly
3. Include: description, reproduction steps, potential impact
4. We will respond within 48 hours

## Security considerations

- **API keys** are stored in `~/.fangbot/.env` with `chmod 600` permissions
- **No real patient data** — synthetic data only (per Resolucao CNS 510/2016)
- **MCP communication** is local stdio only — no network exposure
- **Audit trail** is append-only JSONL — immutable record of all tool calls
- The agent **never computes clinical scores from LLM knowledge** — always delegates to deterministic MCP tools
