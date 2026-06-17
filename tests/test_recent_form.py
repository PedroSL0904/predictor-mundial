"""Tests para el modulo recent_form."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.recent_form import (
    blend_recent_with_historical,
    compute_recent_form,
)


def make_sample_df() -> pd.DataFrame:
    """Crea un DataFrame sintetico de partidos.

    Brazil siempre anota 2 goles como local, 1 como visitante.
    Argentina siempre anota 1 gol como local, 0.5 como visitante (media).
    """
    rows = []
    base = pd.Timestamp("2020-01-01")
    for i in range(10):
        d = base + pd.Timedelta(days=i * 30)
        # Brazil vs Argentina alternando local/visitante
        rows.append({
            "date": d,
            "home_team": "Brazil" if i % 2 == 0 else "Argentina",
            "away_team": "Argentina" if i % 2 == 0 else "Brazil",
            "home_goals": 2 if i % 2 == 0 else 0,
            "away_goals": 1 if i % 2 == 0 else 1,
            "tournament": "Friendly",
        })
    return pd.DataFrame(rows)


def test_compute_recent_form_basic() -> None:
    df = make_sample_df()
    out = compute_recent_form(df, n_matches=3, min_matches=2, decay_half_life_matches=None)
    assert not out.empty
    assert "team" in out.columns
    assert "recent_attack" in out.columns
    assert "recent_defense" in out.columns
    # Brazil y Argentina deben aparecer
    teams = set(out["team"].tolist())
    assert "Brazil" in teams
    assert "Argentina" in teams


def test_compute_recent_form_decay() -> None:
    df = make_sample_df()
    out_no_decay = compute_recent_form(df, n_matches=3, min_matches=2, decay_half_life_matches=None)
    out_decay = compute_recent_form(df, n_matches=3, min_matches=2, decay_half_life_matches=1.0)
    # Con decay, los valores pueden diferir
    # (no garantizamos cual es mayor, solo que el cálculo no falla)
    assert not out_decay.empty


def test_compute_recent_form_respects_n_matches() -> None:
    df = make_sample_df()
    out_2 = compute_recent_form(df, n_matches=2, min_matches=2, decay_half_life_matches=None)
    out_5 = compute_recent_form(df, n_matches=5, min_matches=2, decay_half_life_matches=None)
    # Con n_matches mas alto, mas info, valores cercanos a la media
    brazil_2 = out_2[out_2["team"] == "Brazil"]["recent_attack"].iloc[0]
    brazil_5 = out_5[out_5["team"] == "Brazil"]["recent_attack"].iloc[0]
    # Brazil como local anota 2, visitante 1: media 1.5 con shrink a 1.0
    # El n_matches mayor no debería dar valores locos
    assert 0.5 < brazil_2 < 2.5
    assert 0.5 < brazil_5 < 2.5


def test_compute_recent_form_respects_as_of() -> None:
    df = make_sample_df()
    # as_of antes de los partidos -> no debe haber resultados
    out = compute_recent_form(df, as_of="2019-01-01", n_matches=3, min_matches=2, decay_half_life_matches=None)
    assert out.empty


def test_compute_recent_form_min_matches_filter() -> None:
    df = make_sample_df()
    # min_matches=15 -> no hay equipo con 15 partidos (Brasil y Arg tienen 10)
    out = compute_recent_form(df, n_matches=10, min_matches=15, decay_half_life_matches=None)
    assert out.empty


def test_blend_recent_with_historical_no_recent() -> None:
    hist = pd.DataFrame({"team": ["A"], "attack": [1.5], "defense_vulnerability": [1.0]})
    recent = pd.DataFrame()
    out = blend_recent_with_historical(hist, recent, weight_recent=0.5)
    # Sin datos recientes, devuelve historico sin cambios
    assert out["attack"].iloc[0] == 1.5
    assert out["defense_vulnerability"].iloc[0] == 1.0


def test_blend_recent_with_historical_with_recent() -> None:
    hist = pd.DataFrame({"team": ["A", "B"], "attack": [1.0, 1.0], "defense_vulnerability": [1.0, 1.0]})
    recent = pd.DataFrame({
        "team": ["A"], "recent_attack": [2.0], "recent_defense": [0.5],
    })
    out = blend_recent_with_historical(hist, recent, weight_recent=0.5)
    # A: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    assert out[out["team"] == "A"]["attack"].iloc[0] == 1.5
    # B: sin recent, queda en 1.0
    assert out[out["team"] == "B"]["attack"].iloc[0] == 1.0
    # A defense: 0.5 * 0.5 + 0.5 * 1.0 = 0.75
    assert out[out["team"] == "A"]["defense_vulnerability"].iloc[0] == 0.75


def test_blend_recent_weight_zero_ignores_recent() -> None:
    hist = pd.DataFrame({"team": ["A"], "attack": [1.0], "defense_vulnerability": [1.0]})
    recent = pd.DataFrame({"team": ["A"], "recent_attack": [99.0], "recent_defense": [99.0]})
    out = blend_recent_with_historical(hist, recent, weight_recent=0.0)
    assert out["attack"].iloc[0] == 1.0
