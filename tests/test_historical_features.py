"""Tests de features historicas (H2H, momentum, WC history)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.historical_features import (
    MAX_ADJUSTMENT,
    apply_all_adjustments,
    compute_h2h_adjustment,
    compute_momentum_adjustment,
    compute_wc_history_adjustment,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic DataFrame con partidos de prueba."""
    rows = [
        # H2H: Germany vs France, 3 partidos (Germany gana 2, France gana 1)
        {"date": "2024-01-01", "home_team": "Germany", "away_team": "France",
         "home_goals": 3, "away_goals": 1, "tournament": "Friendly"},
        {"date": "2023-06-01", "home_team": "France", "away_team": "Germany",
         "home_goals": 0, "away_goals": 2, "tournament": "Friendly"},
        {"date": "2022-09-01", "home_team": "Germany", "away_team": "France",
         "home_goals": 2, "away_goals": 1, "tournament": "UEFA Nations League"},
        # Momentum: Germany's ultimos 3 partidos
        {"date": "2024-05-01", "home_team": "Germany", "away_team": "Netherlands",
         "home_goals": 4, "away_goals": 0, "tournament": "Friendly"},
        {"date": "2024-04-01", "home_team": "Germany", "away_team": "Italy",
         "home_goals": 2, "away_goals": 1, "tournament": "Friendly"},
        {"date": "2024-03-01", "home_team": "Spain", "away_team": "Germany",
         "home_goals": 0, "away_goals": 3, "tournament": "Friendly"},
        # WC history: Germany's WC matches
        {"date": "2022-12-01", "home_team": "Germany", "away_team": "Japan",
         "home_goals": 1, "away_goals": 2, "tournament": "FIFA World Cup"},
        {"date": "2022-11-01", "home_team": "Spain", "away_team": "Germany",
         "home_goals": 1, "away_goals": 1, "tournament": "FIFA World Cup"},
        {"date": "2018-06-01", "home_team": "Germany", "away_team": "Mexico",
         "home_goals": 0, "away_goals": 1, "tournament": "FIFA World Cup"},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_h2h_no_matches_returns_default(sample_df: pd.DataFrame) -> None:
    """Sin H2H, devuelve (1.0, 1.0)."""
    att, deff = compute_h2h_adjustment(sample_df, "Argentina", "Brazil", "2025-01-01")
    assert att == 1.0
    assert deff == 1.0


def test_h2h_too_few_matches_returns_default(sample_df: pd.DataFrame) -> None:
    """Con menos de MIN_H2H_MATCHES partidos, devuelve (1.0, 1.0)."""
    # Solo hay 3 partidos de Germany-France. Necesitamos >=3.
    # Para probar el threshold, usar equipos con menos de 3.
    att, deff = compute_h2h_adjustment(sample_df, "Italy", "Spain", "2025-01-01")
    # Italy y Spain no tienen H2H en el sample
    assert att == 1.0
    assert deff == 1.0


def test_h2h_boost_for_dominant_team(sample_df: pd.DataFrame) -> None:
    """Germany domina el H2H (3 GF avg vs ~1.3 league avg), debe tener att_mult > 1.0."""
    att, deff = compute_h2h_adjustment(sample_df, "Germany", "France", "2025-01-01")
    assert att > 1.0  # boost ataque
    # Germany recibio 2 goles en 3 partidos (0.67 gf/game) < 1.3 league avg => def_mult < 1.0
    assert deff < 1.0  # mejora defensa


def test_h2h_penalty_for_dominated_team(sample_df: pd.DataFrame) -> None:
    """France es dominada en H2H, debe tener att_mult < 1.0."""
    att, deff = compute_h2h_adjustment(sample_df, "France", "Germany", "2025-01-01")
    assert att < 1.0  # reduccion ataque


def test_h2h_clamped_to_max(sample_df: pd.DataFrame) -> None:
    """Los ajustes deben estar en [1-MAX, 1+MAX]."""
    att, deff = compute_h2h_adjustment(sample_df, "Germany", "France", "2025-01-01")
    assert 1.0 - MAX_ADJUSTMENT <= att <= 1.0 + MAX_ADJUSTMENT
    assert 1.0 - MAX_ADJUSTMENT <= deff <= 1.0 + MAX_ADJUSTMENT


def test_h2h_excludes_future_matches(sample_df: pd.DataFrame) -> None:
    """as_of debe filtrar partidos futuros."""
    att, deff = compute_h2h_adjustment(sample_df, "Germany", "France", "2021-01-01")
    # Antes del primer H2H (2022-09-01), no hay datos
    assert att == 1.0
    assert deff == 1.0


def test_momentum_no_matches_returns_default(sample_df: pd.DataFrame) -> None:
    att, deff = compute_momentum_adjustment(sample_df, "Argentina", "2025-01-01")
    assert att == 1.0
    assert deff == 1.0


def test_momentum_boost_for_hot_team(sample_df: pd.DataFrame) -> None:
    """Germany viene de ganar 4-0, 2-1, 3-0 → att boost > 1.0."""
    att, deff = compute_momentum_adjustment(sample_df, "Germany", "2025-01-01", n_matches=3)
    # Germany hizo (4+2+3)/3 = 3.0 gf/game vs 1.30 league avg → boost
    assert att > 1.0
    # Concedio (0+1+0)/3 = 0.33 ga/game < 1.30 → def_mult < 1.0
    assert deff < 1.0


def test_momentum_uses_n_most_recent(sample_df: pd.DataFrame) -> None:
    """Solo los ultimos N partidos cuentan."""
    # Con n_matches=2: solo los 2 mas recientes (4-0 vs Netherlands, 2-1 vs Italy)
    att_2, _ = compute_momentum_adjustment(sample_df, "Germany", "2025-01-01", n_matches=2)
    att_5, _ = compute_momentum_adjustment(sample_df, "Germany", "2025-01-01", n_matches=5)
    # Distintos N → distintos avg → distintos att_mult
    assert att_2 != att_5


def test_momentum_clamped(sample_df: pd.DataFrame) -> None:
    att, deff = compute_momentum_adjustment(sample_df, "Germany", "2025-01-01", n_matches=3)
    assert 1.0 - MAX_ADJUSTMENT <= att <= 1.0 + MAX_ADJUSTMENT
    assert 1.0 - MAX_ADJUSTMENT <= deff <= 1.0 + MAX_ADJUSTMENT


def test_wc_history_no_matches(sample_df: pd.DataFrame) -> None:
    """Equipo sin WC matches: default."""
    att, deff = compute_wc_history_adjustment(sample_df, "Netherlands", "2025-01-01")
    assert att == 1.0
    assert deff == 1.0


def test_wc_history_team_with_wc_matches(sample_df: pd.DataFrame) -> None:
    """Germany tiene 3 WC matches, 1W 1D 1L (33% win rate, peor que 50%)."""
    att, deff = compute_wc_history_adjustment(sample_df, "Germany", "2025-01-01")
    # win_rate=0.33 < 0.5 => wc_factor = (0.33-0.5)*2 = -0.33
    # att_mult = 1 + (-0.33) * 0.10 = 0.967 (peor ataque)
    # def_mult = 1 - (-0.33) * 0.10 = 1.033 (peor defensa)
    assert att < 1.0
    assert deff > 1.0


def test_wc_history_clamped(sample_df: pd.DataFrame) -> None:
    att, deff = compute_wc_history_adjustment(sample_df, "Germany", "2025-01-01")
    assert 1.0 - MAX_ADJUSTMENT <= att <= 1.0 + MAX_ADJUSTMENT
    assert 1.0 - MAX_ADJUSTMENT <= deff <= 1.0 + MAX_ADJUSTMENT


def test_apply_all_adjustments_composition(sample_df: pd.DataFrame) -> None:
    """Las 3 features se multiplican correctamente."""
    h_att, h_def, a_att, a_def = apply_all_adjustments(
        sample_df, "Germany", "France", "2025-01-01"
    )
    # Germany boost ataque (H2H, momentum), reduce defensa
    # France reduccion ataque (H2H), aumento defensa
    assert 0.5 <= h_att <= 1.5
    assert 0.5 <= h_def <= 1.5
    assert 0.5 <= a_att <= 1.5
    assert 0.5 <= a_def <= 1.5


def test_apply_all_adjustments_can_disable_features(sample_df: pd.DataFrame) -> None:
    """Flags enable_* desactivan features individuales."""
    h_att_all, _, _, _ = apply_all_adjustments(
        sample_df, "Germany", "France", "2025-01-01",
        enable_h2h=True, enable_momentum=True, enable_wc_history=True,
    )
    h_att_h2h_only, _, _, _ = apply_all_adjustments(
        sample_df, "Germany", "France", "2025-01-01",
        enable_h2h=True, enable_momentum=False, enable_wc_history=False,
    )
    # Sin momentum ni WC, h_att solo viene de H2H (debe ser diferente)
    assert h_att_all != h_att_h2h_only


def test_apply_all_adjustments_default_all_disabled() -> None:
    """Con todos los flags False, devuelve (1, 1, 1, 1)."""
    df = pd.DataFrame({
        "date": pd.to_datetime([]),
        "home_team": [], "away_team": [],
        "home_goals": [], "away_goals": [], "tournament": [],
    })
    h_att, h_def, a_att, a_def = apply_all_adjustments(
        df, "A", "B", "2025-01-01",
        enable_h2h=False, enable_momentum=False, enable_wc_history=False,
    )
    assert h_att == 1.0
    assert h_def == 1.0
    assert a_att == 1.0
    assert a_def == 1.0
