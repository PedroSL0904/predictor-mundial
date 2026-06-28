"""Tests para src/domain.py (incluye outcome_from_score consolidado)."""
import pytest

from src.domain import MatchOutcome, outcome_from_score, outcome_from_score_str


def test_home_win():
    assert outcome_from_score(2, 0) == MatchOutcome.HOME
    assert outcome_from_score(2, 0).value == "H"


def test_away_win():
    assert outcome_from_score(0, 2) == MatchOutcome.AWAY
    assert outcome_from_score(0, 2).value == "A"


def test_draw():
    assert outcome_from_score(1, 1) == MatchOutcome.DRAW
    assert outcome_from_score(1, 1).value == "D"


def test_zero_zero_is_draw():
    assert outcome_from_score(0, 0) == MatchOutcome.DRAW


def test_high_scoring():
    assert outcome_from_score(5, 3) == MatchOutcome.HOME
    assert outcome_from_score(2, 7) == MatchOutcome.AWAY


def test_str_variant():
    assert outcome_from_score_str("2-1") == MatchOutcome.HOME
    assert outcome_from_score_str("0-3") == MatchOutcome.AWAY
    assert outcome_from_score_str("1-1") == MatchOutcome.DRAW


def test_str_invalid_raises():
    with pytest.raises(ValueError):
        outcome_from_score_str("invalid")
    with pytest.raises(ValueError):
        outcome_from_score_str("2-1-0")
