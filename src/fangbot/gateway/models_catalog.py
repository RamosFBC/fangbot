"""Catalog of available models per provider for interactive selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    description: str
    category: str  # e.g. "flagship", "reasoning", "fast", "legacy"


# ---------------------------------------------------------------------------
# OpenAI models
# ---------------------------------------------------------------------------

OPENAI_MODELS: list[ModelInfo] = [
    # GPT-5 family
    ModelInfo("gpt-5", "GPT-5", "Most capable GPT model", "flagship"),
    ModelInfo("gpt-5-mini", "GPT-5 Mini", "Smaller GPT-5 variant, fast and capable", "fast"),
    ModelInfo("gpt-5-nano", "GPT-5 Nano", "Smallest GPT-5 variant, lowest cost", "fast"),
    ModelInfo(
        "gpt-5.4-pro", "GPT-5.4 Pro", "Latest snapshot — smarter and more precise", "flagship"
    ),
    # Codex
    ModelInfo("gpt-5.1-codex", "GPT-5.1 Codex", "Optimized for agentic coding tasks", "coding"),
    ModelInfo("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini", "Smaller coding model", "coding"),
    # GPT-4.1 family
    ModelInfo("gpt-4.1", "GPT-4.1", "Strong general-purpose model, fine-tunable", "general"),
    ModelInfo("gpt-4.1-mini", "GPT-4.1 Mini", "Smaller GPT-4.1 variant", "fast"),
    ModelInfo("gpt-4.1-nano", "GPT-4.1 Nano", "Smallest GPT-4.1 variant", "fast"),
    # GPT-4o family
    ModelInfo("gpt-4o", "GPT-4o", "Multimodal model, good balance of speed and quality", "general"),
    ModelInfo("gpt-4o-mini", "GPT-4o Mini", "Fast and affordable multimodal model", "fast"),
    # Reasoning models (o-series)
    ModelInfo("o3", "o3", "Most powerful reasoning model — coding, math, science", "reasoning"),
    ModelInfo("o3-pro", "o3 Pro", "Extended thinking for most reliable responses", "reasoning"),
    ModelInfo("o3-mini", "o3 Mini", "Fast reasoning, on par with o1 at lower latency", "reasoning"),
    ModelInfo(
        "o4-mini", "o4 Mini", "Cost-efficient reasoning, best on AIME benchmarks", "reasoning"
    ),
    ModelInfo("o1", "o1", "Previous-gen reasoning model", "reasoning"),
    ModelInfo("o1-mini", "o1 Mini", "Previous-gen small reasoning model", "reasoning"),
]

# ---------------------------------------------------------------------------
# Anthropic Claude models
# ---------------------------------------------------------------------------

CLAUDE_MODELS: list[ModelInfo] = [
    # Claude 4.6 family (latest)
    ModelInfo(
        "claude-opus-4-6",
        "Claude Opus 4.6",
        "Most intelligent — coding, agents, complex tasks",
        "flagship",
    ),
    ModelInfo(
        "claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "Balanced speed and intelligence, fewer tokens",
        "general",
    ),
    # Claude 4.5 family
    ModelInfo(
        "claude-haiku-4-5-20251001",
        "Claude Haiku 4.5",
        "Fast and affordable for everyday tasks",
        "fast",
    ),
    # Claude 4 family
    ModelInfo(
        "claude-sonnet-4-20250514", "Claude Sonnet 4", "Previous-gen balanced model", "general"
    ),
    ModelInfo(
        "claude-opus-4-20250514", "Claude Opus 4", "Previous-gen most capable model", "flagship"
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROVIDER_MODELS: dict[str, list[ModelInfo]] = {
    "openai": OPENAI_MODELS,
    "claude": CLAUDE_MODELS,
}

PROVIDER_DEFAULTS: dict[str, str] = {
    "claude": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}

CATEGORY_STYLES: dict[str, str] = {
    "flagship": "bold magenta",
    "general": "cyan",
    "fast": "green",
    "reasoning": "bold yellow",
    "coding": "bold blue",
    "legacy": "dim",
}
