"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMAGENT_", env_file=".env")

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

    @property
    def mcp_args_list(self) -> list[str]:
        return self.mcp_args.split(",")


def get_settings() -> Settings:
    return Settings()
