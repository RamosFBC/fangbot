"""Pre-round summary workflow — problem-based summary with trends and baselines."""

from __future__ import annotations

from fangbot.chart.models import FactCategory, FactStatus
from fangbot.chart.temporal import classify_facts, compare_to_baseline
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


class ClassifyTemporal(WorkflowStep):
    name = "classify_temporal"
    step_type = StepType.ANALYZE

    async def execute(self, context: WorkflowContext) -> StepResult:
        temporal_facts = classify_facts(context.chart)
        classifications = [
            {
                "name": tf.fact.name,
                "classification": tf.classification.value,
                "rationale": tf.rationale,
            }
            for tf in temporal_facts
        ]
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"classifications": classifications},
            provenance=[
                f"{tf.fact.name}: {tf.classification.value}" for tf in temporal_facts
            ],
            confidence=0.9,
            duration_ms=0,
        )


class DetectTrendsStep(WorkflowStep):
    name = "detect_trends"
    step_type = StepType.ANALYZE

    async def execute(self, context: WorkflowContext) -> StepResult:
        trends = detect_trends(context.chart)
        trend_summaries = [
            {"name": t.fact_name, "direction": t.direction.value, "summary": t.summary}
            for t in trends
        ]
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"trends": trend_summaries},
            provenance=[t.summary for t in trends if t.summary],
            confidence=0.95 if trends else 0.7,
            duration_ms=0,
        )


class CompareBaseline(WorkflowStep):
    name = "compare_baseline"
    step_type = StepType.ANALYZE

    async def execute(self, context: WorkflowContext) -> StepResult:
        comparisons = compare_to_baseline(context.chart)
        comparison_data = [
            {
                "name": c.fact_name,
                "baseline": c.baseline_value,
                "current": c.current_value,
                "change_percent": c.change_percent,
                "summary": c.summary,
            }
            for c in comparisons
        ]
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"comparisons": comparison_data},
            provenance=[c.summary for c in comparisons if c.summary],
            confidence=0.95 if comparisons else 0.7,
            duration_ms=0,
        )


class GenerateSummary(WorkflowStep):
    name = "generate_summary"
    step_type = StepType.GENERATE

    async def execute(self, context: WorkflowContext) -> StepResult:
        parts: list[str] = []

        # Active problems
        active_dx = [
            f for f in context.chart.facts
            if f.category == FactCategory.DIAGNOSIS and f.status == FactStatus.ACTIVE
        ]
        historical_dx = [
            f for f in context.chart.facts
            if f.category == FactCategory.DIAGNOSIS and f.status == FactStatus.HISTORICAL
        ]
        if active_dx:
            parts.append("Active problems: " + ", ".join(f.value for f in active_dx))
        if historical_dx:
            parts.append("Background: " + ", ".join(f.value for f in historical_dx))

        # Trends
        trends_result = context.step_results.get("detect_trends")
        if trends_result and trends_result.output.get("trends"):
            trend_lines = [
                t["summary"] for t in trends_result.output["trends"] if t["summary"]
            ]
            if trend_lines:
                parts.append("Trends:\n" + "\n".join(f"  - {t}" for t in trend_lines))

        # Baseline comparisons
        baseline_result = context.step_results.get("compare_baseline")
        if baseline_result and baseline_result.output.get("comparisons"):
            comp_lines = [
                c["summary"]
                for c in baseline_result.output["comparisons"]
                if c["summary"]
            ]
            if comp_lines:
                parts.append(
                    "Baseline comparisons:\n"
                    + "\n".join(f"  - {c}" for c in comp_lines)
                )

        # Temporal classifications
        temporal_result = context.step_results.get("classify_temporal")
        if temporal_result and temporal_result.output.get("classifications"):
            new_items = [
                c["name"] for c in temporal_result.output["classifications"]
                if c["classification"] in ("new", "worsening")
            ]
            if new_items:
                parts.append(f"New/worsening: {', '.join(new_items)}")

        # Latest labs and vitals
        labs = [f for f in context.chart.facts if f.category == FactCategory.LAB]
        vitals = [
            f for f in context.chart.facts
            if f.category == FactCategory.VITAL
            and f.name.lower() not in ("age", "sex", "gender")
        ]
        active_meds = [
            f for f in context.chart.facts
            if f.category == FactCategory.MEDICATION and f.status == FactStatus.ACTIVE
        ]

        if labs:
            latest_labs: dict[str, str] = {}
            for lab in sorted(labs, key=lambda f: f.timestamp or f.confidence):
                latest_labs[lab.name] = lab.value
            parts.append(
                "Latest labs: "
                + ", ".join(f"{k}: {v}" for k, v in latest_labs.items())
            )

        if vitals:
            parts.append(
                "Vitals: " + ", ".join(f"{f.name}: {f.value}" for f in vitals)
            )

        if active_meds:
            parts.append(
                "Active meds: "
                + ", ".join(f"{f.name} {f.value}" for f in active_meds)
            )

        summary = "\n\n".join(parts) if parts else context.raw_text

        prompt = (
            "Generate a concise, problem-based pre-round summary. Structure as:\n"
            "1. Active Problems (with latest relevant data and trends)\n"
            "2. Chronic/Background conditions\n"
            "3. Overnight Events (notable changes)\n"
            "4. Suggested Plan (brief next steps)\n\n"
            "Use only the data provided. Highlight worsening trends.\n\n"
            f"Clinical data:\n{summary}"
        )

        response = await context.provider.call(
            messages=[Message(role=Role.USER, content=prompt)],
            system=(
                "You are a clinical note drafting assistant. "
                "Generate problem-based pre-round summaries."
            ),
        )

        all_provenance: list[str] = []
        for step_result in context.step_results.values():
            all_provenance.extend(step_result.provenance)

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"summary": response.content},
            provenance=list(set(all_provenance)),
            confidence=min(
                (r.confidence for r in context.step_results.values()),
                default=0.5,
            ),
            duration_ms=0,
        )


class PreRoundSummary(BaseWorkflow):
    name = "pre_round_summary"
    description = (
        "Generate a problem-based pre-round summary with trends, "
        "baselines, and temporal analysis"
    )
    input_description = (
        "Patient chart with labs, vitals, medications across multiple time points"
    )
    output_description = (
        "Problem-based summary organized by active problems, trends, "
        "and suggested plan"
    )

    def steps(self) -> list[WorkflowStep]:
        return [
            ClassifyTemporal(),
            DetectTrendsStep(),
            CompareBaseline(),
            GenerateSummary(),
        ]

    def build_draft(self, context: WorkflowContext) -> WorkflowDraft:
        gen_result = context.step_results.get("generate_summary")
        warnings: list[str] = []

        if gen_result and gen_result.status == StepStatus.COMPLETED:
            content = gen_result.output.get("summary", "")
            provenance = gen_result.provenance
        else:
            content = "Unable to generate pre-round summary."
            provenance = []
            warnings.append("Summary generation failed")

        for result in context.step_results.values():
            if result.status == StepStatus.FAILED:
                warnings.append(f"Step '{result.step_name}' failed: {result.error}")

        confidences = [r.confidence for r in context.step_results.values()]
        overall = min(confidences) if confidences else 0.0

        return WorkflowDraft(
            workflow_name=self.name,
            sections=[
                DraftSection(
                    heading="Pre-Round Summary",
                    content=content,
                    provenance=provenance,
                    confidence=overall,
                ),
            ],
            step_results=list(context.step_results.values()),
            overall_confidence=overall,
            warnings=warnings,
        )
