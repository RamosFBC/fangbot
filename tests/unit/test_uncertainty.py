"""Tests for the uncertainty calibration module."""

from __future__ import annotations

import pytest

from fangbot.brain.uncertainty import (
    ConfidenceLevel,
    UncertaintyAssessment,
    parse_uncertainty_assessment,
    strip_uncertainty_block,
)


class TestConfidenceLevel:
    def test_enum_values(self) -> None:
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MODERATE.value == "moderate"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.INSUFFICIENT_DATA.value == "insufficient_data"

    def test_numeric_value_high(self) -> None:
        assert ConfidenceLevel.HIGH.numeric_value == 1.0

    def test_numeric_value_moderate(self) -> None:
        assert ConfidenceLevel.MODERATE.numeric_value == 0.7

    def test_numeric_value_low(self) -> None:
        assert ConfidenceLevel.LOW.numeric_value == 0.4

    def test_numeric_value_insufficient(self) -> None:
        assert ConfidenceLevel.INSUFFICIENT_DATA.numeric_value == 0.0


class TestUncertaintyAssessment:
    def test_escalation_recommended_for_low(self) -> None:
        assessment = UncertaintyAssessment(
            confidence=ConfidenceLevel.LOW,
            reasoning="Missing lab values",
        )
        assert assessment.escalation_recommended is True

    def test_escalation_not_recommended_for_high(self) -> None:
        assessment = UncertaintyAssessment(
            confidence=ConfidenceLevel.HIGH,
            reasoning="All data present",
        )
        assert assessment.escalation_recommended is False


class TestParseUncertaintyAssessment:
    def test_complete_block_parsed(self) -> None:
        text = (
            "The score is 3.\n"
            "---\n"
            "Confidence: MODERATE\n"
            "Reasoning: Age was estimated\n"
            "Missing data: Exact DOB; Smoking status\n"
            "Contradictions: None\n"
            "Would change answer: Exact DOB\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.confidence == ConfidenceLevel.MODERATE
        assert result.reasoning == "Age was estimated"
        assert result.missing_data == ["Exact DOB", "Smoking status"]
        assert result.contradictions == []
        assert result.would_change == ["Exact DOB"]

    def test_high_confidence_no_missing(self) -> None:
        text = (
            "Result done.\n"
            "---\n"
            "Confidence: HIGH\n"
            "Reasoning: All parameters present and validated\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "Would change answer: None\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.missing_data == []
        assert result.contradictions == []
        assert result.would_change == []
        assert result.escalation_recommended is False

    def test_multiple_items_in_lists(self) -> None:
        text = (
            "---\n"
            "Confidence: LOW\n"
            "Reasoning: Several gaps\n"
            "Missing data: Creatinine; Weight; Height\n"
            "Contradictions: BP 120/80 vs BP 140/90; Age 65 vs Age 70\n"
            "Would change answer: Creatinine; Weight\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert len(result.missing_data) == 3
        assert len(result.contradictions) == 2
        assert result.would_change == ["Creatinine", "Weight"]

    def test_contradictions_present(self) -> None:
        text = (
            "---\n"
            "Confidence: LOW\n"
            "Reasoning: Conflicting values\n"
            "Missing data: None\n"
            "Contradictions: Creatinine 1.2 vs 2.4\n"
            "Would change answer: Creatinine\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.contradictions == ["Creatinine 1.2 vs 2.4"]
        assert result.would_change == ["Creatinine"]

    def test_insufficient_data_level(self) -> None:
        text = (
            "---\n"
            "Confidence: INSUFFICIENT_DATA\n"
            "Reasoning: No labs available\n"
            "Missing data: All lab values\n"
            "Contradictions: None\n"
            "Would change answer: All lab values\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.confidence == ConfidenceLevel.INSUFFICIENT_DATA
        assert result.escalation_recommended is True

    def test_returns_none_when_no_block(self) -> None:
        text = "The CHA2DS2-VASc score is 3. No uncertainty block here."
        result = parse_uncertainty_assessment(text)
        assert result is None

    def test_strip_uncertainty_block_removes_block(self) -> None:
        text = (
            "The score is 3.\n"
            "---\n"
            "Confidence: HIGH\n"
            "Reasoning: Complete data\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "Would change answer: None\n"
            "---"
        )
        stripped = strip_uncertainty_block(text)
        assert "Confidence" not in stripped
        assert "The score is 3." in stripped

    def test_strip_preserves_text_without_block(self) -> None:
        text = "Just a normal response."
        assert strip_uncertainty_block(text) == text

    def test_strip_removes_trailing_whitespace(self) -> None:
        text = (
            "Result here.\n\n"
            "---\n"
            "Confidence: HIGH\n"
            "Reasoning: OK\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "Would change answer: None\n"
            "---\n"
        )
        stripped = strip_uncertainty_block(text)
        assert stripped == "Result here."


class TestUncertaintyEdgeCases:
    def test_extra_whitespace_in_block(self) -> None:
        text = (
            "---\n"
            "Confidence:   HIGH  \n"
            "Reasoning:   All data present  \n"
            "Missing data:   None  \n"
            "Contradictions:   None  \n"
            "Would change answer:   None  \n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.reasoning == "All data present"

    def test_lowercase_confidence_level(self) -> None:
        text = (
            "---\n"
            "Confidence: moderate\n"
            "Reasoning: Some inference needed\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "Would change answer: None\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.confidence == ConfidenceLevel.MODERATE

    def test_invalid_confidence_level_returns_none(self) -> None:
        text = (
            "---\n"
            "Confidence: UNKNOWN_LEVEL\n"
            "Reasoning: Bad level\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "Would change answer: None\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is None

    def test_multiple_semicolon_items_with_whitespace(self) -> None:
        text = (
            "---\n"
            "Confidence: LOW\n"
            "Reasoning: Gaps\n"
            "Missing data:  Creatinine ;  Weight ;  Height \n"
            "Contradictions: BP conflict ; Age conflict \n"
            "Would change answer: Creatinine ; Weight \n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.missing_data == ["Creatinine", "Weight", "Height"]
        assert result.contradictions == ["BP conflict", "Age conflict"]
        assert result.would_change == ["Creatinine", "Weight"]

    def test_multiline_synthesis_preserved_after_strip(self) -> None:
        text = (
            "Line one.\n"
            "Line two.\n"
            "Line three.\n"
            "---\n"
            "Confidence: HIGH\n"
            "Reasoning: OK\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "Would change answer: None\n"
            "---"
        )
        stripped = strip_uncertainty_block(text)
        assert "Line one." in stripped
        assert "Line two." in stripped
        assert "Line three." in stripped
        assert "Confidence" not in stripped

    def test_numeric_values_ordering(self) -> None:
        assert ConfidenceLevel.HIGH.numeric_value > ConfidenceLevel.MODERATE.numeric_value
        assert ConfidenceLevel.MODERATE.numeric_value > ConfidenceLevel.LOW.numeric_value
        assert ConfidenceLevel.LOW.numeric_value > ConfidenceLevel.INSUFFICIENT_DATA.numeric_value

    def test_serialization_round_trip(self) -> None:
        assessment = UncertaintyAssessment(
            confidence=ConfidenceLevel.MODERATE,
            reasoning="Age estimated from context",
            missing_data=["Exact DOB", "Smoking status"],
            contradictions=["BP 120/80 vs 140/90"],
            would_change=["Exact DOB"],
        )
        json_str = assessment.model_dump_json()
        restored = UncertaintyAssessment.model_validate_json(json_str)
        assert restored.confidence == ConfidenceLevel.MODERATE
        assert restored.reasoning == "Age estimated from context"
        assert restored.missing_data == ["Exact DOB", "Smoking status"]
        assert restored.contradictions == ["BP 120/80 vs 140/90"]
        assert restored.would_change == ["Exact DOB"]
        assert restored.escalation_recommended is False

    def test_would_change_parsed_correctly(self) -> None:
        text = (
            "---\n"
            "Confidence: LOW\n"
            "Reasoning: Key data missing\n"
            "Missing data: Creatinine; eGFR\n"
            "Contradictions: None\n"
            "Would change answer: Creatinine; eGFR; Medication list\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.would_change == ["Creatinine", "eGFR", "Medication list"]

    def test_would_change_none_returns_empty_list(self) -> None:
        text = (
            "---\n"
            "Confidence: HIGH\n"
            "Reasoning: All present\n"
            "Missing data: None\n"
            "Contradictions: None\n"
            "Would change answer: None\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.would_change == []

    def test_none_detected_filtered(self) -> None:
        text = (
            "---\n"
            "Confidence: HIGH\n"
            "Reasoning: All present\n"
            "Missing data: None detected\n"
            "Contradictions: None detected\n"
            "Would change answer: None detected\n"
            "---"
        )
        result = parse_uncertainty_assessment(text)
        assert result is not None
        assert result.missing_data == []
        assert result.contradictions == []
        assert result.would_change == []
