"""Tests for the uncertainty calibration module."""

from __future__ import annotations

import pytest

from fangbot.brain.uncertainty import ConfidenceLevel, UncertaintyAssessment


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
