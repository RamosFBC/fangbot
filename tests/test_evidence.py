"""Tests for evidence retrieval orchestrator."""

from __future__ import annotations

from fangbot.skills.evidence import (
    EvidenceCitation,
    EvidenceConflict,
    EvidenceExtractor,
    EvidenceSource,
    EvidenceStrength,
    GuidelineReference,
)


class TestEvidenceModels:
    """Test evidence data models."""

    def test_evidence_citation_creation(self):
        citation = EvidenceCitation(
            doi="10.1161/CIR.0000000000001123",
            source=EvidenceSource.GUIDELINE,
            strength=EvidenceStrength.STRONG,
            recommendation="Anticoagulation recommended for CHA2DS2-VASc >= 2 in men",
            guideline_id="aha-acc-hrs-2023-afib",
            section="4.1.1 Stroke Prevention",
            organization="AHA/ACC/HRS",
        )
        assert citation.doi == "10.1161/CIR.0000000000001123"
        assert citation.source == EvidenceSource.GUIDELINE
        assert citation.strength == EvidenceStrength.STRONG
        assert citation.section == "4.1.1 Stroke Prevention"

    def test_evidence_citation_minimal(self):
        """Citation with only required fields."""
        citation = EvidenceCitation(
            recommendation="Use CKD-EPI for GFR estimation",
            source=EvidenceSource.GUIDELINE,
        )
        assert citation.doi is None
        assert citation.strength is None
        assert citation.section is None

    def test_guideline_reference_creation(self):
        ref = GuidelineReference(
            guideline_id="kdigo-2024-ckd",
            title="KDIGO 2024 Clinical Practice Guideline for CKD",
            organization="KDIGO",
            year=2024,
            sections_consulted=["Chapter 1: Definition and Classification"],
        )
        assert ref.guideline_id == "kdigo-2024-ckd"
        assert ref.year == 2024
        assert len(ref.sections_consulted) == 1

    def test_evidence_conflict_creation(self):
        citation_a = EvidenceCitation(
            recommendation="Target BP < 130/80",
            source=EvidenceSource.GUIDELINE,
            organization="AHA/ACC",
            doi="10.1161/HYP.001",
        )
        citation_b = EvidenceCitation(
            recommendation="Target BP < 140/90 for most patients",
            source=EvidenceSource.GUIDELINE,
            organization="ESC/ESH",
            doi="10.1093/eurheartj/ehy339",
        )
        conflict = EvidenceConflict(
            topic="Blood pressure target in hypertension",
            citations=[citation_a, citation_b],
            description="AHA/ACC recommends <130/80 while ESC/ESH recommends <140/90 for most patients",
        )
        assert len(conflict.citations) == 2
        assert "Blood pressure" in conflict.topic

    def test_evidence_source_enum(self):
        assert EvidenceSource.GUIDELINE.value == "guideline"
        assert EvidenceSource.LANDMARK_TRIAL.value == "landmark_trial"
        assert EvidenceSource.HOSPITAL_PROTOCOL.value == "hospital_protocol"

    def test_evidence_strength_enum(self):
        assert EvidenceStrength.STRONG.value == "strong"
        assert EvidenceStrength.MODERATE.value == "moderate"
        assert EvidenceStrength.WEAK.value == "weak"
        assert EvidenceStrength.EXPERT_OPINION.value == "expert_opinion"


class TestEvidenceExtractor:
    """Test evidence extraction from MCP tool result text."""

    def setup_method(self):
        self.extractor = EvidenceExtractor()

    def test_extract_dois_from_text(self):
        text = (
            "Guideline: AHA/ACC 2023 Atrial Fibrillation Guidelines\n"
            "DOI: 10.1161/CIR.0000000000001123\n"
            "Recommendation: Oral anticoagulation is recommended for patients with AF "
            "and a CHA2DS2-VASc score >= 2 in men or >= 3 in women."
        )
        citations = self.extractor.extract_citations(text, tool_name="retrieve_guideline")
        assert len(citations) >= 1
        assert any(c.doi == "10.1161/CIR.0000000000001123" for c in citations)

    def test_extract_pmids_from_text(self):
        text = "PMID: 37634560\nThe SPRINT trial showed intensive BP control reduced events."
        citations = self.extractor.extract_citations(text, tool_name="retrieve_guideline")
        assert any(c.pmid == "37634560" for c in citations)

    def test_extract_guideline_reference(self):
        text = (
            "Guideline ID: kdigo-2024-ckd\n"
            "Title: KDIGO 2024 Clinical Practice Guideline for CKD Evaluation and Management\n"
            "Organization: KDIGO\n"
            "Year: 2024\n"
            "Section: Chapter 1 — Definition and Classification of CKD"
        )
        ref = self.extractor.extract_guideline_reference(text)
        assert ref is not None
        assert ref.guideline_id == "kdigo-2024-ckd"
        assert ref.organization == "KDIGO"
        assert ref.year == 2024

    def test_extract_from_search_results(self):
        text = (
            "Found 2 guidelines:\n"
            "1. [aha-afib-2023] AHA/ACC/HRS 2023 AFib Guidelines — DOI: 10.1161/CIR.001\n"
            "2. [esc-afib-2020] ESC 2020 AFib Guidelines — DOI: 10.1093/eurheartj/ehaa612"
        )
        refs = self.extractor.extract_search_results(text)
        assert len(refs) == 2
        assert refs[0].guideline_id == "aha-afib-2023"
        assert refs[1].guideline_id == "esc-afib-2020"

    def test_classify_source_type_guideline(self):
        text = (
            "Guideline: AHA/ACC 2023\n"
            "Recommendation Class I, Level of Evidence A\n"
            "DOI: 10.1161/CIR.0000000000001123"
        )
        citations = self.extractor.extract_citations(text, tool_name="retrieve_guideline")
        assert len(citations) >= 1
        assert all(c.source == EvidenceSource.GUIDELINE for c in citations)

    def test_classify_source_type_trial(self):
        text = "Landmark Trial: SPRINT (2015)\nPMID: 26551272\nIntensive BP control reduced CV events."
        citations = self.extractor.extract_citations(text, tool_name="retrieve_guideline")
        assert any(c.source == EvidenceSource.LANDMARK_TRIAL for c in citations)

    def test_extract_strength(self):
        text = (
            "Recommendation: Use direct oral anticoagulants over warfarin.\n"
            "Strength: Strong recommendation (Class I, Level A)\n"
            "DOI: 10.1161/CIR.001"
        )
        citations = self.extractor.extract_citations(text, tool_name="retrieve_guideline")
        assert len(citations) >= 1
        assert citations[0].strength is not None

    def test_empty_text_returns_no_citations(self):
        citations = self.extractor.extract_citations("", tool_name="retrieve_guideline")
        assert citations == []

    def test_no_evidence_in_text(self):
        citations = self.extractor.extract_citations(
            "The weather is nice today.", tool_name="retrieve_guideline"
        )
        assert citations == []

    def test_extract_search_results_empty(self):
        refs = self.extractor.extract_search_results("No guidelines found.")
        assert refs == []
