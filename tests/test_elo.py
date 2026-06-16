"""Tests del sistema Elo y cálculo de strengths ponderados."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data.elo import (
    BASE_K,
    ORIGINAL_ELO,
    EloRatingSystem,
    current_top_n,
)
from src.features.strengths import (
    _approx_xg_from_elo,
    build_elo_lookup_at,
    compute_weighted_strengths,
)


def test_initial_rating_is_1500() -> None:
    elo = EloRatingSystem()
    assert elo.get_rating("Argentina") == 1500.0
    assert elo.get_rating("Brazil") == 1500.0


def test_winner_gains_rating() -> None:
    elo = EloRatingSystem()
    elo.update("Argentina", "Brazil", 2, 0, neutral=True, date="2024-01-01")
    assert elo.get_rating("Argentina") > 1500
    assert elo.get_rating("Brazil") < 1500


def test_loser_loses_rating() -> None:
    elo = EloRatingSystem()
    elo.update("Brazil", "Argentina", 0, 2, neutral=True, date="2024-01-01")
    assert elo.get_rating("Brazil") < 1500
    assert elo.get_rating("Argentina") > 1500


def test_draw_moves_ratings_slightly() -> None:
    elo = EloRatingSystem()
    # Argentina 1500 vs Brazil 1500, empate: el cambio debe ser 0
    elo.update("Argentina", "Brazil", 1, 1, neutral=True, date="2024-01-01")
    assert math.isclose(elo.get_rating("Argentina"), 1500.0, abs_tol=1e-9)
    assert math.isclose(elo.get_rating("Brazil"), 1500.0, abs_tol=1e-9)


def test_bigger_underdog_winner_gains_more() -> None:
    """Si un equipo mucho más débil le gana a uno fuerte, gana más rating."""
    # Caso 1: un equipo más débil (Bottom) le gana a uno fuerte (Top)
    elo_a = EloRatingSystem()
    # Pre-seteamos ratings para que haya gap
    elo_a.set_rating("Top", 1700.0)
    elo_a.set_rating("Bottom", 1300.0)
    elo_a.update("Top", "Bottom", 0, 1, neutral=True, date="2024-01-01")
    delta_a = elo_a.get_rating("Bottom") - 1300

    # Caso 2: un equipo más fuerte (Top) le gana a uno débil (Bottom)
    elo_b = EloRatingSystem()
    elo_b.set_rating("Top", 1700.0)
    elo_b.set_rating("Bottom", 1300.0)
    elo_b.update("Top", "Bottom", 1, 0, neutral=True, date="2024-01-01")
    delta_b = elo_b.get_rating("Top") - 1700

    # El underdog ganando debe ganar más rating que el favorito ganando
    assert delta_a > delta_b


def test_home_advantage_increases_expected() -> None:
    elo = EloRatingSystem()
    # Argentina 1500 vs Brazil 1500, no neutral: Argentina tiene ventaja
    e_home = elo.expected_score(1500, 1500, neutral=False)
    e_neutral = elo.expected_score(1500, 1500, neutral=True)
    assert e_home > e_neutral


def test_margin_of_victory_multiplier_increases_k() -> None:
    elo_no_mov = EloRatingSystem(mov_multiplier=False)
    elo_no_mov.update("A", "B", 0, 4, neutral=True, date="2024-01-01")
    delta_no_mov = elo_no_mov.get_rating("B") - 1500

    elo_with_mov = EloRatingSystem(mov_multiplier=True)
    elo_with_mov.update("A", "B", 0, 4, neutral=True, date="2024-01-01")
    delta_with_mov = elo_with_mov.get_rating("B") - 1500

    # Con MOV, B ganó con más diferencia → más rating
    assert delta_with_mov > delta_no_mov


def test_process_dataframe_in_order() -> None:
    df = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 0, "neutral_venue": True, "tournament": "Friendly"},
        {"date": pd.Timestamp("2024-02-01"), "home_team": "B", "away_team": "A",
         "home_goals": 0, "away_goals": 2, "neutral_venue": True, "tournament": "Friendly"},
    ])
    elo = EloRatingSystem()
    elo.process_dataframe(df)
    # A ganó el segundo partido como visitante → debe estar arriba
    assert elo.get_rating("A") > elo.get_rating("B")


def test_process_dataframe_skips_nan_goals() -> None:
    df = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 0, "neutral_venue": True, "tournament": "Friendly"},
        {"date": pd.Timestamp("2024-02-01"), "home_team": "B", "away_team": "A",
         "home_goals": None, "away_goals": None, "neutral_venue": True, "tournament": "Future"},
    ])
    elo = EloRatingSystem()
    elo.process_dataframe(df)  # no debe fallar
    # Solo se procesó el primer partido
    assert elo.get_rating("A") > 1500


def test_approx_xg_from_elo() -> None:
    # Mismo Elo: 1.30
    xg = _approx_xg_from_elo(1500, 1500)
    assert math.isclose(xg, 1.30, abs_tol=0.01)
    # Atacante muy superior: > 1.30
    assert _approx_xg_from_elo(1900, 1500) > 1.30
    # Atacante muy inferior: < 1.30
    assert _approx_xg_from_elo(1300, 1500) < 1.30


def test_weighted_strengths_uses_elo() -> None:
    """Verifica que el ponderador por Elo funciona."""
    # Equipo A juega contra 5 rivales débiles (Elo 1300) y marca muchos goles
    # Equipo A también juega contra 1 rival fuerte (Elo 1900) y marca pocos
    # Sin ponderación, ataque inflado. Con ponderación, debe ser más bajo.
    df = pd.DataFrame([
        # A vs débiles: muchos goles
        {"date": pd.Timestamp("2024-01-01"), "home_team": "A", "away_team": f"Weak{i}",
         "home_goals": 4, "away_goals": 0, "neutral_venue": True, "tournament": "Friendly"}
        for i in range(5)
    ] + [
        # A vs fuerte: pocos goles
        {"date": pd.Timestamp("2024-01-01"), "home_team": "A", "away_team": "Strong",
         "home_goals": 0, "away_goals": 2, "neutral_venue": True, "tournament": "Friendly"},
    ])

    elo_lookup = {f"Weak{i}": 1300.0 for i in range(5)}
    elo_lookup["Strong"] = 1900.0
    elo_lookup["A"] = 1500.0

    s = compute_weighted_strengths(
        df, elo_lookup=elo_lookup, elo_sigma=200.0,
        min_weighted_matches=3.0, shrinkage_matches=0,
    )
    attack_a = float(s[s["team"] == "A"]["attack"].iloc[0])
    # El ataque de A debe ser < 4.0 (lo que marcaría sin ponderar)
    # porque los 4 goles vs débiles se ponderaron a la baja
    assert attack_a < 4.0


def test_strong_team_has_high_attack_with_weighting() -> None:
    """Un equipo que le mete goles a rivales FUERTES debe tener attack > 1.0"""
    df = pd.DataFrame([
        # TopA le mete 2-3 goles a TopB (Elo similar a TopA)
        {"date": pd.Timestamp("2024-01-01"), "home_team": "TopA", "away_team": "TopB",
         "home_goals": 2, "away_goals": 1, "neutral_venue": True, "tournament": "Friendly"},
        {"date": pd.Timestamp("2024-02-01"), "home_team": "TopB", "away_team": "TopA",
         "home_goals": 0, "away_goals": 3, "neutral_venue": True, "tournament": "Friendly"},
    ])
    elo_lookup = {"TopA": 1900.0, "TopB": 1880.0}
    s = compute_weighted_strengths(
        df, elo_lookup=elo_lookup, elo_sigma=300.0,
        min_weighted_matches=1.0, shrinkage_matches=0,
    )
    attack_a = float(s[s["team"] == "TopA"]["attack"].iloc[0])
    # Marcó 5 goles en 2 partidos contra TopB (Elo similar)
    # xG esperado contra rival similar ≈ 1.30 por partido
    # attack = 5 / (1.30 * 2) ≈ 1.92
    assert attack_a > 1.5
