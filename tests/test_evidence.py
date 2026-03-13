"""Tests for evidence retrieval orchestrator."""

from __future__ import annotations

from fangbot.skills.evidence import (
    EvidenceCitation,
    EvidenceConflict,
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
