"""Batch runner for evaluation studies — runs gold standard cases through the agent."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fangbot.evaluation.models import CaseResult, GoldStandardCase, RiskTier, StudyConfig

logger = logging.getLogger(__name__)


def _extract_score_from_synthesis(synthesis: str) -> int | float | None:
    """Attempt to extract a numeric score from agent synthesis text."""
    patterns = [
        r"score[:\s]+(\d+)",
        r"score\s+(?:is|of|=)\s+(\d+)",
        r"CHA2DS2-VASc[:\s]+(\d+)",
        r"GCS[:\s]+(\d+)",
        r"total[:\s]+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, synthesis, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_tier_from_synthesis(synthesis: str) -> RiskTier | None:
    """Attempt to extract risk tier from agent synthesis text."""
    lower = synthesis.lower()
    if "high" in lower and "risk" in lower:
        return RiskTier.HIGH
    if "moderate" in lower and "risk" in lower:
        return RiskTier.MODERATE
    if "low" in lower and "risk" in lower:
        return RiskTier.LOW
    for tier in [RiskTier.HIGH, RiskTier.MODERATE, RiskTier.LOW]:
        if tier.value in lower:
            return tier
    return None


class BatchRunner:
    """Runs gold standard cases through the clinical agent for evaluation."""

    def __init__(self, config: StudyConfig):
        self._config = config

    async def run_cases(
        self,
        cases: list[GoldStandardCase],
        provider_name: str,
        model_name: str,
    ) -> list[CaseResult]:
        """Run all cases and return results."""
        results: list[CaseResult] = []
        for case in cases:
            logger.info(f"Running case {case.case_id}...")
            result = await self._run_single_case(case, provider_name, model_name)
            results.append(result)
        return results

    async def _run_single_case(
        self,
        case: GoldStandardCase,
        provider_name: str,
        model_name: str,
    ) -> CaseResult:
        """Run a single case through the full ReAct agent pipeline.

        This is the live integration point — requires MCP server and LLM provider.
        For unit tests, this method is mocked.
        """
        from fangbot.brain.react import ReActLoop
        from fangbot.brain.system_prompt import CLINICAL_SYSTEM_PROMPT
        from fangbot.config import get_settings
        from fangbot.memory.audit import AuditLogger
        from fangbot.memory.session import SessionContext
        from fangbot.skills.mcp_client import OpenMedicineMCPClient
        from fangbot.skills.tool_registry import ToolRegistry

        settings = get_settings()
        audit = AuditLogger(log_dir=settings.log_dir)
        session_id = audit.start_session()

        start = time.monotonic()

        try:
            provider = _create_provider(provider_name, model_name)
            mcp = OpenMedicineMCPClient(
                command=settings.mcp_command,
                args=settings.mcp_args_list,
            )
            async with mcp.connect():
                registry = ToolRegistry(mcp)
                tools = await registry.get_tools()
                session = SessionContext(system_prompt=CLINICAL_SYSTEM_PROMPT)
                react = ReActLoop(
                    provider=provider,
                    mcp_client=mcp,
                    audit_logger=audit,
                    max_iterations=self._config.max_iterations,
                )
                result = await react.run(case.narrative, session, tools)

            duration = time.monotonic() - start
            return CaseResult(
                case_id=case.case_id,
                provider=provider_name,
                model=model_name,
                actual_score=_extract_score_from_synthesis(result.synthesis),
                actual_risk_tier=_extract_tier_from_synthesis(result.synthesis),
                actual_tool_calls=result.tool_calls_made,
                synthesis=result.synthesis,
                chain_of_thought=result.chain_of_thought,
                guardrail_passed=result.guardrail_passed,
                iterations=result.iterations,
                duration_seconds=duration,
                audit_session_id=session_id,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            logger.error(f"Case {case.case_id} failed: {exc}")
            return CaseResult(
                case_id=case.case_id,
                provider=provider_name,
                model=model_name,
                error=str(exc),
                duration_seconds=duration,
                audit_session_id=session_id,
            )

    def save_results(
        self,
        results: list[CaseResult],
        provider_name: str,
        model_name: str,
    ) -> Path:
        """Save results to JSON file in the results directory."""
        results_dir = Path(self._config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_model = model_name.replace("/", "_")
        filename = f"{provider_name}_{safe_model}_{timestamp}.json"
        filepath = results_dir / filename

        data = {
            "study_name": self._config.study_name,
            "provider": provider_name,
            "model": model_name,
            "timestamp": timestamp,
            "cases": [r.model_dump(mode="json") for r in results],
        }
        filepath.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Results saved to {filepath}")
        return filepath


def _create_provider(provider_name: str, model_name: str) -> Any:
    """Create an LLM provider instance by name."""
    from fangbot.config import get_settings

    settings = get_settings()

    if provider_name == "claude":
        from fangbot.brain.providers.claude import ClaudeProvider

        return ClaudeProvider(api_key=settings.anthropic_api_key, model=model_name)
    elif provider_name == "openai":
        from fangbot.brain.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=settings.openai_api_key, model=model_name)
    elif provider_name == "local":
        from fangbot.brain.providers.local import LocalProvider

        return LocalProvider(
            base_url=settings.local_base_url,
            api_key=settings.local_api_key,
            model=model_name,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
