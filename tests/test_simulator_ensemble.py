"""Tests de integracion del Ensemble en TournamentSimulator (Sprint A7)."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.features.strengths_cache import StrengthsCache
from src.models import (
    BivariatePoissonModel,
    EnsembleModel,
    PoissonGoalModel,
    SkellamModel,
)
from src.simulation.wc2026_simulate import TournamentSimulator


@pytest.fixture(scope="module")
def loaded() -> tuple[pd.DataFrame, dict, StrengthsCache]:
    """Carga CSV + timeline + cache una sola vez para todos los tests."""
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.parquet")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)
    return df, timeline, cache


def _build_ensemble() -> EnsembleModel:
    """Ensemble con pesos uniformes para testing."""
    s = {
        "draw_boost": 0.10,
        "draw_penalty_threshold": 0.08,
        "draw_penalty_strength": 0.05,
        "elo_gap_inflation": 0.30,
        "league_avg_multiplier": 1.18,
    }
    return EnsembleModel(
        [
            PoissonGoalModel(**s),
            BivariatePoissonModel(lambda_3=0.10, **s),
            SkellamModel(**s),
        ],
        weights=[1/3, 1/3, 1/3],
    )


def test_default_uses_poisson(loaded: tuple[pd.DataFrame, dict, StrengthsCache]) -> None:
    """Sin model=... debe usar PoissonGoalModel (backward compat)."""
    df, timeline, cache = loaded
    sim = TournamentSimulator(df, timeline, cache, as_of="2026-06-10")
    assert isinstance(sim.model, PoissonGoalModel)


def test_ensemble_model_accepted(loaded: tuple[pd.DataFrame, dict, StrengthsCache]) -> None:
    """model=EnsembleModel(...) debe aceptarse sin error."""
    df, timeline, cache = loaded
    ens = _build_ensemble()
    sim = TournamentSimulator(df, timeline, cache, as_of="2026-06-10", model=ens)
    assert sim.model is ens


def test_predict_works_with_ensemble(loaded: tuple[pd.DataFrame, dict, StrengthsCache]) -> None:
    """TournamentSimulator.predict() funciona con EnsembleModel."""
    df, timeline, cache = loaded
    ens = _build_ensemble()
    sim = TournamentSimulator(df, timeline, cache, as_of="2026-06-10", model=ens)
    pred = sim.predict("Brazil", "Argentina")
    assert pred is not None
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_predict_ensemble_matches_ensemble_only(
    loaded: tuple[pd.DataFrame, dict, StrengthsCache]
) -> None:
    """Las probs del sim con ensemble deben coincidir con EnsembleModel directo."""
    df, timeline, cache = loaded
    ens = _build_ensemble()
    sim = TournamentSimulator(df, timeline, cache, as_of="2026-06-10", model=ens)

    # Hacer una predicción via sim
    sim_pred = sim.predict("Germany", "Japan")

    # Construir el ensemble directo y comparar
    # (mismas strengths por construcción del sim)
    from src.config import get_settings
    s = get_settings()
    common = {
        "draw_boost": s.draw_boost,
        "draw_penalty_threshold": s.draw_penalty_threshold,
        "draw_penalty_strength": s.draw_penalty_strength,
        "elo_gap_inflation": s.elo_gap_inflation,
        "league_avg_multiplier": 1.18,
    }
    direct_ens = EnsembleModel(
        [
            PoissonGoalModel(**common),
            BivariatePoissonModel(lambda_3=0.10, **common),
            SkellamModel(**common),
        ],
        weights=[1/3, 1/3, 1/3],
    )
    from src.data.elo import ORIGINAL_ELO
    from src.data.team_names import OLO_TO_MARTJ
    from src.models import TeamStrength

    home_martj = OLO_TO_MARTJ.get("Germany", "Germany")
    away_martj = OLO_TO_MARTJ.get("Japan", "Japan")
    h = sim.strength_by_team.loc[home_martj]
    a = sim.strength_by_team.loc[away_martj]
    home = TeamStrength(
        name=home_martj,
        attack=float(h["attack"]),
        defense_vulnerability=float(h["defense_vulnerability"]),
    )
    away = TeamStrength(
        name=away_martj,
        attack=float(a["attack"]),
        defense_vulnerability=float(a["defense_vulnerability"]),
    )
    home_elo = sim.elo_lookup.get(home_martj, ORIGINAL_ELO)
    away_elo = sim.elo_lookup.get(away_martj, ORIGINAL_ELO)
    direct_pred = direct_ens.predict(home, away, home_elo=home_elo, away_elo=away_elo)
    assert math.isclose(sim_pred.p_home, direct_pred.p_home, abs_tol=1e-6)
    assert math.isclose(sim_pred.p_draw, direct_pred.p_draw, abs_tol=1e-6)
    assert math.isclose(sim_pred.p_away, direct_pred.p_away, abs_tol=1e-6)


def test_pure_poisson_ensemble_backward_compat(
    loaded: tuple[pd.DataFrame, dict, StrengthsCache]
) -> None:
    """EnsembleModel([P, BP, S], weights=[1, 0, 0]) debe dar mismo resultado
    que TournamentSimulator default (Poisson puro)."""
    df, timeline, cache = loaded
    sim_default = TournamentSimulator(df, timeline, cache, as_of="2026-06-10")
    s = {
        "draw_boost": 0.10,
        "draw_penalty_threshold": 0.08,
        "draw_penalty_strength": 0.05,
        "elo_gap_inflation": 0.30,
        "league_avg_multiplier": 1.18,
    }
    pure_p_ensemble = EnsembleModel(
        [
            PoissonGoalModel(**s),
            BivariatePoissonModel(lambda_3=0.10, **s),
            SkellamModel(**s),
        ],
        weights=[1.0, 0.0, 0.0],
    )
    sim_ensemble = TournamentSimulator(
        df, timeline, cache, as_of="2026-06-10", model=pure_p_ensemble
    )

    p_default = sim_default.predict("Brazil", "Argentina")
    p_ensemble = sim_ensemble.predict("Brazil", "Argentina")
    assert math.isclose(p_default.p_home, p_ensemble.p_home, abs_tol=1e-9)
    assert math.isclose(p_default.p_draw, p_ensemble.p_draw, abs_tol=1e-9)
    assert math.isclose(p_default.p_away, p_ensemble.p_away, abs_tol=1e-9)
