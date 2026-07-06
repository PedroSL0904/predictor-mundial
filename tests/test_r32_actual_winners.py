"""Tests para extract_r32_actual_winners."""
import pandas as pd

from src.simulation.wc2026_simulate import extract_r32_actual_winners


def _make_r32_df():
    """Crea un mini DataFrame con partidos R32 jugados."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-07-01", "2026-07-02", "2026-07-02", "2026-07-03",
            "2026-07-03", "2026-07-04", "2026-07-04", "2026-07-04",
        ]),
        "home_team": [
            "South Africa", "Germany", "Netherlands", "France",
            "Ivory Coast", "Mexico", "England", "United States",
        ],
        "away_team": [
            "Canada", "Paraguay", "Morocco", "Sweden",
            "Norway", "Ecuador", "DR Congo", "Bosnia and Herzegovina",
        ],
        "home_goals": [0, 3, 1, 3, 1, 2, 2, 2],
        "away_goals": [1, 1, 1, 0, 2, 0, 1, 0],
        "tournament": ["FIFA World Cup"] * 8,
        "city": ["H"] * 8,
        "country": ["USA"] * 8,
        "neutral_venue": [True] * 8,
    })


def test_extract_winners_basic():
    df = _make_r32_df()
    winners = extract_r32_actual_winners(df)
    # 8 partidos conocidos, todos deberian mapear a tie_ids 73-80
    assert 73 in winners
    assert 74 in winners
    assert 80 in winners
    # Winners correctos segun scores
    assert winners[73] == "Canada"  # 0-1, away won
    assert winners[74] == "Germany"  # 3-1, home won
    assert winners[75] == "Morocco"  # 1-1, real winner via penalty override (NED-MAR pens)
    assert 76 not in winners  # not in our test data
    assert winners[77] == "France"  # 3-0, home won
    assert winners[78] == "Norway"  # 1-2, away won
    assert winners[79] == "Mexico"  # 2-0, home won
    assert winners[80] == "England"  # 2-1, home won


def test_extract_winners_unknown_matchup():
    df = _make_r32_df()
    # Agregar partido con matchup no conocido
    extra = pd.DataFrame({
        "date": pd.to_datetime(["2026-07-08"]),
        "home_team": ["X"],
        "away_team": ["Y"],
        "home_goals": [1],
        "away_goals": [0],
        "tournament": ["FIFA World Cup"],
        "city": [""],
        "country": [""],
        "neutral_venue": [True],
    })
    df = pd.concat([df, extra], ignore_index=True)
    winners = extract_r32_actual_winners(df)
    # Partido desconocido no debe aparecer
    assert 99 not in winners
    # Los conocidos siguen funcionando
    assert winners[73] == "Canada"


def test_extract_winners_empty():
    """Si no hay R32 en el DataFrame, retorna dict vacio."""
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
    winners = extract_r32_actual_winners(df)
    assert winners == {}


def test_simulate_with_r32_actual_winners():
    """Verifica que simulate_tournament acepta el parametro r32_actual_winners."""
    import inspect

    from src.simulation.wc2026_simulate import simulate_tournament

    sig = inspect.signature(simulate_tournament)
    assert "r32_actual_winners" in sig.parameters
    assert sig.parameters["r32_actual_winners"].default is None
