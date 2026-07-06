"""Tests para extract_r16_actual_winners."""
import pandas as pd

from src.simulation.wc2026_simulate import extract_r16_actual_winners


def _make_r16_df():
    """Crea un mini DataFrame con partidos R16 jugados."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-07-04", "2026-07-04", "2026-07-05", "2026-07-05",
        ]),
        "home_team": [
            "Paraguay", "Canada", "Brazil", "Mexico",
        ],
        "away_team": [
            "France", "Morocco", "Norway", "England",
        ],
        "home_goals": [0, 0, 1, 2],
        "away_goals": [1, 3, 2, 3],
        "tournament": ["FIFA World Cup"] * 4,
        "city": ["H"] * 4,
        "country": ["USA"] * 4,
        "neutral_venue": [True] * 4,
    })


def test_extract_r16_winners_basic():
    df = _make_r16_df()
    winners = extract_r16_actual_winners(df)
    # 4 partidos conocidos
    assert 89 in winners
    assert 90 in winners
    assert 91 in winners
    assert 92 in winners
    # Winners correctos
    assert winners[89] == "France"  # 0-1, away
    assert winners[90] == "Morocco"  # 0-3, away
    assert winners[91] == "Norway"  # 1-2, away (UPSET)
    assert winners[92] == "England"  # 2-3, away


def test_extract_r16_winners_empty():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01"]),
        "home_team": ["A"],
        "away_team": ["B"],
        "home_goals": [1.0],
        "away_goals": [0.0],
        "tournament": ["FIFA World Cup"],
        "city": [""],
        "country": [""],
        "neutral_venue": [True],
    })
    assert extract_r16_actual_winners(df) == {}


def test_extract_r16_winners_unknown_matchup():
    df = _make_r16_df()
    extra = pd.DataFrame({
        "date": pd.to_datetime(["2026-07-06"]),
        "home_team": ["X"],
        "away_team": ["Y"],
        "home_goals": [1.0],
        "away_goals": [0.0],
        "tournament": ["FIFA World Cup"],
        "city": [""],
        "country": [""],
        "neutral_venue": [True],
    })
    df = pd.concat([df, extra], ignore_index=True)
    winners = extract_r16_actual_winners(df)
    assert 99 not in winners
    assert winners[91] == "Norway"  # known matchup still works
