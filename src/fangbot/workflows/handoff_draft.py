"""Handoff draft workflow — generates IPASS-format handoff notes."""

from __future__ import annotations

from fangbot.chart.consistency import run_all_checks
from fangbot.chart.models import FactCategory, FactStatus
from fangbot.chart.trends import detect_trends
from fangbot.models import Message, Role
from fangbot.workflows.engine import BaseWorkflow, WorkflowContext, WorkflowStep
from fangbot.workflows.models import (
    DraftSection,
    StepResult,
    StepStatus,
    StepType,
    WorkflowDraft,
)


class IdentifyActiveProblems(WorkflowStep):
    name = "identify_active_problems"
    step_type = StepType.EXTRACT

    async def execute(self, context: WorkflowContext) -> StepResult:
        problems: list[dict[str, str]] = []
        provenance: list[str] = []
        for fact in context.chart.facts:
            if fact.category == FactCategory.DIAGNOSIS and fact.status == FactStatus.ACTIVE:
                problems.append({"name": fact.name, "detail": fact.value})
                provenance.append(fact.source)
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"problems": problems},
            provenance=provenance,
            confidence=0.9 if problems else 0.3,
            duration_ms=0,
        )


class AnalyzeTrends(WorkflowStep):
    name = "analyze_trends"
    step_type = StepType.ANALYZE

    async def execute(self, context: WorkflowContext) -> StepResult:
        trends = detect_trends(context.chart)
        trend_summaries = [
            {"name": t.fact_name, "direction": t.direction.value, "summary": t.summary}
            for t in trends
        ]
        provenance = [f"{t.fact_name}: {t.summary}" for t in trends]
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"trends": trend_summaries},
            provenance=provenance,
            confidence=0.95 if trends else 0.7,
            duration_ms=0,
        )


class CheckConsistency(WorkflowStep):
    name = "check_consistency"
    step_type = StepType.VALIDATE

    async def execute(self, context: WorkflowContext) -> StepResult:
        report = run_all_checks(context.chart)
        issues = [
            {"type": i.type.value, "severity": i.severity.value, "description": i.description}
            for i in report.inconsistencies
        ]
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"issues": issues, "is_clean": report.is_clean},
            provenance=[i.description for i in report.inconsistencies],
            confidence=1.0,
            duration_ms=0,
        )


class GenerateIPASS(WorkflowStep):
    name = "generate_ipass"
    step_type = StepType.GENERATE

    async def execute(self, context: WorkflowContext) -> StepResult:
        parts: list[str] = []

        problems_result = context.step_results.get("identify_active_problems")
        if problems_result and problems_result.output.get("problems"):
            problems_text = ", ".join(
                p["name"] for p in problems_result.output["problems"]
            )
            parts.append(f"Active problems: {problems_text}")

        trends_result = context.step_results.get("analyze_trends")
        if trends_result and trends_result.output.get("trends"):
            trend_lines = [
                t["summary"] for t in trends_result.output["trends"] if t["summary"]
            ]
            if trend_lines:
                parts.append(f"Trends: {'; '.join(trend_lines)}")

        consistency_result = context.step_results.get("check_consistency")
        if consistency_result and not consistency_result.output.get("is_clean", True):
            issues = consistency_result.output.get("issues", [])
            issue_lines = [i["description"] for i in issues]
            parts.append(f"Chart issues: {'; '.join(issue_lines)}")

        active_meds = [
            f"{f.name} {f.value}" for f in context.chart.facts
            if f.category == FactCategory.MEDICATION and f.status == FactStatus.ACTIVE
        ]
        if active_meds:
            parts.append(f"Active medications: {', '.join(active_meds)}")

        summary = "\n".join(parts) if parts else context.raw_text

        prompt = (
            "Generate a clinical handoff note in IPASS format:\n"
            "I — Illness Severity (stable/watcher/unstable)\n"
            "P — Patient Summary (one-liner + key overnight events)\n"
            "A — Action List (pending tasks, if-then contingencies)\n"
            "S — Situation Awareness (what to watch for)\n"
            "S — Synthesis by Receiver (confirm understanding)\n\n"
            "Use only the data provided. Do not invent facts.\n\n"
            f"Clinical data:\n{summary}"
        )

        response = await context.provider.call(
            messages=[Message(role=Role.USER, content=prompt)],
            system=(
                "You are a clinical note drafting assistant. "
                "Generate accurate, structured handoff notes."
            ),
        )

        all_provenance: list[str] = []
        for step_result in context.step_results.values():
            all_provenance.extend(step_result.provenance)

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"ipass": response.content},
            provenance=list(set(all_provenance)),
            confidence=min(
                (r.confidence for r in context.step_results.values()),
                default=0.5,
            ),
            duration_ms=0,
        )


class HandoffDraft(BaseWorkflow):
    name = "handoff_draft"
    description = (
        "Generate an IPASS-format clinical handoff note "
        "with trends and consistency checks"
    )
    input_description = (
        "Patient chart with diagnoses, labs, vitals, medications "
        "over multiple time points"
    )
    output_description = (
        "IPASS handoff note (Illness severity, Patient summary, "
        "Action list, Situation awareness, Synthesis)"
    )

    def steps(self) -> list[WorkflowStep]:
        return [
            IdentifyActiveProblems(),
            AnalyzeTrends(),
            CheckConsistency(),
            GenerateIPASS(),
        ]

    def build_draft(self, context: WorkflowContext) -> WorkflowDraft:
        gen_result = context.step_results.get("generate_ipass")
        warnings: list[str] = []

        if gen_result and gen_result.status == StepStatus.COMPLETED:
            content = gen_result.output.get("ipass", "")
            provenance = gen_result.provenance
        else:
            content = "Unable to generate handoff — generation step failed."
            provenance = []
            warnings.append("IPASS generation failed")

        consistency_result = context.step_results.get("check_consistency")
        if consistency_result and not consistency_result.output.get("is_clean", True):
            for issue in consistency_result.output.get("issues", []):
                warnings.append(f"Chart issue: {issue['description']}")

        for result in context.step_results.values():
            if result.status == StepStatus.FAILED:
                warnings.append(f"Step '{result.step_name}' failed: {result.error}")

        confidences = [r.confidence for r in context.step_results.values()]
        overall = min(confidences) if confidences else 0.0

        return WorkflowDraft(
            workflow_name=self.name,
            sections=[
                DraftSection(
                    heading="IPASS Handoff",
                    content=content,
                    provenance=provenance,
                    confidence=overall,
                ),
            ],
            step_results=list(context.step_results.values()),
            overall_confidence=overall,
            warnings=warnings,
        )
