"""Admission one-liner workflow — generates a structured admission summary."""

from __future__ import annotations

from fangbot.chart.models import FactCategory, FactStatus
from fangbot.models import Message, Role
from fangbot.workflows.engine import BaseWorkflow, WorkflowContext, WorkflowStep
from fangbot.workflows.models import (
    DraftSection,
    StepResult,
    StepStatus,
    StepType,
    WorkflowDraft,
)


class ExtractDemographics(WorkflowStep):
    name = "extract_demographics"
    step_type = StepType.EXTRACT

    async def execute(self, context: WorkflowContext) -> StepResult:
        demographics: dict[str, str] = {}
        provenance: list[str] = []
        for fact in context.chart.facts:
            name_lower = fact.name.lower()
            if name_lower in ("age", "sex", "gender"):
                demographics[name_lower] = fact.value
                provenance.append(fact.source)
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output=demographics,
            provenance=provenance,
            confidence=1.0 if demographics else 0.5,
            duration_ms=0,
        )


class ExtractChiefComplaint(WorkflowStep):
    name = "extract_chief_complaint"
    step_type = StepType.EXTRACT

    async def execute(self, context: WorkflowContext) -> StepResult:
        cc_keywords = {"chief complaint", "presenting complaint", "reason for visit"}
        complaint = ""
        provenance: list[str] = []
        for fact in context.chart.facts:
            if fact.name.lower() in cc_keywords or (
                fact.category == FactCategory.DIAGNOSIS
                and fact.status == FactStatus.ACTIVE
                and "chief" in (fact.source_location or "").lower()
            ):
                complaint = fact.value
                provenance.append(f"{fact.source}: {fact.source_location or ''}")
                break
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"chief_complaint": complaint},
            provenance=provenance,
            confidence=0.9 if complaint else 0.3,
            duration_ms=0,
        )


class ExtractKeyFindings(WorkflowStep):
    name = "extract_key_findings"
    step_type = StepType.EXTRACT

    async def execute(self, context: WorkflowContext) -> StepResult:
        pmh: list[str] = []
        labs: list[str] = []
        vitals: list[str] = []
        meds: list[str] = []
        provenance: list[str] = []

        for fact in context.chart.facts:
            if fact.category == FactCategory.DIAGNOSIS and fact.status in (
                FactStatus.ACTIVE, FactStatus.HISTORICAL
            ):
                if "chief" not in (fact.source_location or "").lower():
                    pmh.append(fact.value)
                    provenance.append(fact.source)
            elif fact.category == FactCategory.LAB:
                labs.append(f"{fact.name}: {fact.value}")
                provenance.append(fact.source)
            elif (
                fact.category == FactCategory.VITAL
                and fact.name.lower() not in ("age", "sex", "gender")
            ):
                vitals.append(f"{fact.name}: {fact.value}")
                provenance.append(fact.source)
            elif fact.category == FactCategory.MEDICATION and fact.status == FactStatus.ACTIVE:
                meds.append(f"{fact.name} {fact.value}")
                provenance.append(fact.source)

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"pmh": pmh, "labs": labs, "vitals": vitals, "medications": meds},
            provenance=list(set(provenance)),
            confidence=0.9 if (pmh or labs) else 0.5,
            duration_ms=0,
        )


class GenerateOneLiner(WorkflowStep):
    name = "generate_oneliner"
    step_type = StepType.GENERATE

    async def execute(self, context: WorkflowContext) -> StepResult:
        demographics = context.step_results.get("extract_demographics")
        cc = context.step_results.get("extract_chief_complaint")
        findings = context.step_results.get("extract_key_findings")

        parts: list[str] = []
        if demographics and demographics.output:
            parts.append(f"Demographics: {demographics.output}")
        if cc and cc.output.get("chief_complaint"):
            parts.append(f"Chief complaint: {cc.output['chief_complaint']}")
        if findings and findings.output:
            for key in ("pmh", "labs", "vitals", "medications"):
                items = findings.output.get(key, [])
                if items:
                    parts.append(f"{key.upper()}: {', '.join(items)}")

        extraction_summary = "\n".join(parts) if parts else context.raw_text

        prompt = (
            "Generate a concise admission one-liner in the format: "
            '"[Age][Sex] with [PMH] presenting with [chief complaint] '
            'found to have [key findings] and is being treated with [interventions]." '
            "Use only the data provided. Do not invent facts.\n\n"
            f"Extracted data:\n{extraction_summary}"
        )

        response = await context.provider.call(
            messages=[Message(role=Role.USER, content=prompt)],
            system="You are a clinical note drafting assistant. Generate concise, accurate clinical text.",
        )

        provenance: list[str] = []
        for step_name in ("extract_demographics", "extract_chief_complaint", "extract_key_findings"):
            step_result = context.step_results.get(step_name)
            if step_result:
                provenance.extend(step_result.provenance)

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            status=StepStatus.COMPLETED,
            output={"oneliner": response.content},
            provenance=list(set(provenance)),
            confidence=min(
                (r.confidence for r in context.step_results.values()),
                default=0.5,
            ),
            duration_ms=0,
        )


class AdmissionOneLiner(BaseWorkflow):
    name = "admission_oneliner"
    description = "Generate a structured admission one-liner from H&P note data"
    input_description = "Patient chart with demographics, chief complaint, PMH, labs, vitals"
    output_description = "Single-sentence admission summary in standard format"

    def steps(self) -> list[WorkflowStep]:
        return [
            ExtractDemographics(),
            ExtractChiefComplaint(),
            ExtractKeyFindings(),
            GenerateOneLiner(),
        ]

    def build_draft(self, context: WorkflowContext) -> WorkflowDraft:
        gen_result = context.step_results.get("generate_oneliner")
        warnings: list[str] = []

        if gen_result and gen_result.status == StepStatus.COMPLETED:
            content = gen_result.output.get("oneliner", "")
            provenance = gen_result.provenance
        else:
            content = "Unable to generate one-liner — generation step failed."
            provenance = []
            warnings.append("Generation step failed")

        for result in context.step_results.values():
            if result.confidence < 0.5:
                warnings.append(f"Low confidence in {result.step_name}: {result.confidence:.1f}")
            if result.status == StepStatus.FAILED:
                warnings.append(f"Step '{result.step_name}' failed: {result.error}")

        confidences = [r.confidence for r in context.step_results.values()]
        overall = min(confidences) if confidences else 0.0

        return WorkflowDraft(
            workflow_name=self.name,
            sections=[
                DraftSection(
                    heading="One-Liner",
                    content=content,
                    provenance=provenance,
                    confidence=overall,
                ),
            ],
            step_results=list(context.step_results.values()),
            overall_confidence=overall,
            warnings=warnings,
        )
