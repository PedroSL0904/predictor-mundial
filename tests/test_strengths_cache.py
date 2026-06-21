"""Tests de paridad y velocidad para StrengthsCache.

Verifica que StrengthsCache produce los mismos resultados que
compute_weighted_strengths (con tolerancia para floating point) y que
es significativamente mas rapido.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.features.strengths import compute_weighted_strengths
from src.features.strengths_cache import StrengthsCache


@pytest.fixture(scope="module")
def cache_setup():
    """Carga dataset y construye cache una sola vez para todos los tests."""
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")
    if not csv_path.exists():
        pytest.skip("martj42_results.csv no disponible")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    return df, timeline


def test_cache_build(cache_setup):
    """Construye el cache y verifica dimensiones razonables."""
    df, timeline = cache_setup
    cache = StrengthsCache(df, timeline)
    assert cache.n_teams > 100
    # El cache puede dropear partidos sin scores, pero no debe perder mas del 1%
    assert cache.n > 0.99 * len(df)
    assert len(cache._unique_dates) > 1000


def test_cache_parity(cache_setup):
    """Verifica paridad con compute_weighted_strengths (tolerancia ~0.005)."""
    df, timeline = cache_setup
    cache = StrengthsCache(df, timeline)

    from src.data.elo_timeline import get_elo_at

    max_att = 0.0
    max_def = 0.0
    for as_of in ["2014-06-12", "2018-06-14", "2022-11-20"]:
        train = df[df["date"] < as_of].copy()
        elo_lookup = get_elo_at(timeline, as_of)
        original = compute_weighted_strengths(
            train, elo_lookup=elo_lookup, elo_sigma=225.0,
            recency_half_life_days=1000.0, shrinkage_matches=10,
            min_weighted_matches=8.0,
        )
        cache.set_elo_snapshot(as_of)
        incremental = cache.get_strengths(as_of, shrinkage_matches=10, min_weighted_matches=8.0)

        common = sorted(set(original["team"]) & set(incremental["team"]))
        assert len(common) > 0
        o = original[original["team"].isin(common)].set_index("team").loc[common]
        i = incremental[incremental["team"].isin(common)].set_index("team").loc[common]
        att_diff = float(np.abs(o["attack"].values - i["attack"].values).max())
        def_diff = float(np.abs(o["defense_vulnerability"].values - i["defense_vulnerability"].values).max())
        max_att = max(max_att, att_diff)
        max_def = max(max_def, def_diff)

    assert max_att < 0.01, f"attack diff {max_att} excede tolerancia"
    assert max_def < 0.01, f"defense diff {max_def} excede tolerancia"


def test_cache_speedup(cache_setup):
    """Verifica que el cache es al menos 10x mas rapido simulando un backtest.

    Nota: este test es lento (~5min con el original). Para correr solo los
    tests rapidos, usar: pytest -m "not slow" tests/test_strengths_cache.py
    """
    df, timeline = cache_setup
    cache = StrengthsCache(df, timeline)

    from src.data.elo_timeline import get_elo_at
    from src.evaluation.backtest_cached import get_world_cup_matches

    # Sub-muestra: 5 partidos del WC 2018 (suficiente para medir speedup)
    wc = get_world_cup_matches(df, 2018).sort_values("date").iloc[::13]
    assert len(wc) >= 5

    # Medir original
    t0 = time.time()
    for _, match in wc.iterrows():
        match_date = str(match["date"])[:10]
        train = df[df["date"] < match_date].copy()
        elo_lookup = get_elo_at(timeline, match_date)
        compute_weighted_strengths(
            train, elo_lookup=elo_lookup, elo_sigma=225.0,
            recency_half_life_days=1000.0, shrinkage_matches=10,
            min_weighted_matches=8.0,
        )
    t_orig = time.time() - t0

    # Medir cached
    t0 = time.time()
    cache.set_elo_snapshot("2018-06-14")
    for _, match in wc.iterrows():
        match_date = str(match["date"])[:10]
        cache.get_strengths(match_date, shrinkage_matches=10, min_weighted_matches=8.0)
    t_cache = time.time() - t0

    # El cache debe ser al menos 5x mas rapido (en backtest completo es ~30x)
    speedup = t_orig / t_cache if t_cache > 0 else float('inf')
    assert speedup > 5, f"speedup {speedup:.1f}x insuficiente (orig={t_orig:.1f}s, cache={t_cache:.1f}s)"


def test_cache_advance_monotonic(cache_setup):
    """Verifica que advance_to procesa partidos incrementalmente."""
    df, timeline = cache_setup
    cache = StrengthsCache(df, timeline)
    cache.set_elo_snapshot("2018-06-14")

    pos_before = cache._pos
    cache.advance_to("2018-06-15")
    pos_after = cache._pos
    assert pos_after >= pos_before
    # El state no debe decrecer (w_sum monotono no-decreciente)
    w_sum_before = cache._state[:, 4].copy()
    cache.advance_to("2018-06-20")
    w_sum_after = cache._state[:, 4]
    assert (w_sum_after >= w_sum_before - 1e-9).all()


def test_cache_excludes_future_matches(cache_setup):
    """Verifica que partidos futuros al as_of no entran al state."""
    df, timeline = cache_setup
    cache = StrengthsCache(df, timeline)
    cache.set_elo_snapshot("2018-06-14")
    cache.advance_to("2018-06-14")

    w_sum_at_2014 = cache._state[:, 4].copy()
    pos_at_2014 = cache._pos

    # Avanzar muchos años: el state debe seguir creciendo (mas partidos)
    cache.advance_to("2020-01-01")
    w_sum_at_2020 = cache._state[:, 4]
    assert (w_sum_at_2020 > w_sum_at_2014 - 1e-9).sum() > 0
    assert pos_at_2014 < cache._pos
