"""Application configuration via pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ~/.fangbot/ is the canonical config home (like .claude/, .openclaw/)
FANGBOT_HOME = Path.home() / ".fangbot"

# Load env files: project-local .env first, then ~/.fangbot/.env as fallback.
# SDK-native env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY) become available to all libraries.
load_dotenv()  # project-local .env
load_dotenv(FANGBOT_HOME / ".env")  # user-global config


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FANGBOT_", env_file=".env", extra="ignore")

    # LLM provider
    provider: str = "claude"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    local_base_url: str = ""
    local_api_key: str = ""
    model: str = "claude-sonnet-4-20250514"

    # MCP server — production default assumes open-medicine-mcp is on PATH
    mcp_command: str = "open-medicine-mcp"
    mcp_args: str = ""

    # Agent parameters
    max_iterations: int = 10
    temperature: float = 0.0

    # Logging — defaults to ~/.fangbot/logs/
    log_dir: str = str(FANGBOT_HOME / "logs")
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_api_keys_from_env(self) -> Settings:
        """Fall back to native SDK env var names if FANGBOT_* versions are empty."""
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.google_api_key:
            self.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        return self

    @property
    def mcp_args_list(self) -> list[str]:
        if not self.mcp_args:
            return []
        return self.mcp_args.split(",")


def get_settings() -> Settings:
    return Settings()
