"""Application configuration via pydantic-settings."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into the actual environment so SDK-native env vars
# (OPENAI_API_KEY, ANTHROPIC_API_KEY) are available to all libraries.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FANGBOT_", env_file=".env", extra="ignore")

    # LLM provider
    provider: str = "claude"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    model: str = "claude-sonnet-4-20250514"

    # MCP server
    mcp_command: str = "uv"
    mcp_args: str = "run,open-medicine-mcp"

    # Agent parameters
    max_iterations: int = 10
    temperature: float = 0.0

    # Logging
    log_dir: str = "logs"
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
        return self.mcp_args.split(",")


def get_settings() -> Settings:
    return Settings()
