"""Tests de integracion de features historicas (H2H, momentum, WC) en el simulator."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.features.strengths_cache import StrengthsCache
from src.simulation.wc2026_simulate import TournamentSimulator


@pytest.fixture(scope="module")
def loaded() -> tuple[pd.DataFrame, dict, StrengthsCache]:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.parquet")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)
    return df, timeline, cache


def test_features_disabled_backward_compat(
    loaded: tuple[pd.DataFrame, dict, StrengthsCache]
) -> None:
    """Con enable_historical_features=False, el sim NO aplica H2H/momentum/WC."""
    df, timeline, cache = loaded
    sim = TournamentSimulator(
        df, timeline, cache, as_of="2026-06-10",
        enable_historical_features=False,
    )
    assert sim.enable_historical_features is False


def test_features_enabled_by_default(
    loaded: tuple[pd.DataFrame, dict, StrengthsCache]
) -> None:
    """Por default, enable_historical_features=True (Sprint A5b)."""
    df, timeline, cache = loaded
    sim = TournamentSimulator(df, timeline, cache, as_of="2026-06-10")
    assert sim.enable_historical_features is True


def test_features_change_predictions(
    loaded: tuple[pd.DataFrame, dict, StrengthsCache]
) -> None:
    """Las features (ON vs OFF) producen predicciones distintas."""
    df, timeline, cache = loaded
    sim_off = TournamentSimulator(
        df, timeline, cache, as_of="2026-06-10",
        enable_historical_features=False,
    )
    sim_on = TournamentSimulator(
        df, timeline, cache, as_of="2026-06-10",
        enable_historical_features=True,
    )
    p_off = sim_off.predict("Germany", "Japan")
    p_on = sim_on.predict("Germany", "Japan")
    # Las features pueden o no cambiar la prediccion (depende del match).
    # Alemania y Japon tienen H2H history (Friendly 2024 Germany 4-1 Japan).
    # Alemania deberia tener un att boost por haber ganado bien antes.
    # Las probs no son identicas.
    different = (
        not math.isclose(p_off.p_home, p_on.p_home, abs_tol=1e-6)
        or not math.isclose(p_off.p_draw, p_on.p_draw, abs_tol=1e-6)
        or not math.isclose(p_off.p_away, p_on.p_away, abs_tol=1e-6)
    )
    assert different, "Features ON vs OFF deberia cambiar probs"


def test_features_keep_probabilities_valid(
    loaded: tuple[pd.DataFrame, dict, StrengthsCache]
) -> None:
    """Las features no deben romper probs (siguen sumando 1)."""
    df, timeline, cache = loaded
    sim = TournamentSimulator(
        df, timeline, cache, as_of="2026-06-10",
        enable_historical_features=True,
    )
    for home, away in [("Germany", "Japan"), ("Brazil", "Argentina"),
                        ("France", "England"), ("Spain", "Portugal")]:
        p = sim.predict(home, away)
        total = p.p_home + p.p_draw + p.p_away
        assert math.isclose(total, 1.0, abs_tol=1e-6), f"{home} vs {away}: total={total}"
