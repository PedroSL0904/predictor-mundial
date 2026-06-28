"""Tests del Ensemble model."""
from __future__ import annotations

import math

import pytest

from src.models.bivariate_poisson import BivariatePoissonModel
from src.models.ensemble import EnsembleModel
from src.models.poisson import PoissonGoalModel, TeamStrength
from src.models.skellam import SkellamModel


@pytest.fixture
def strong_home() -> TeamStrength:
    return TeamStrength(name="Germany", attack=2.10, defense_vulnerability=0.85, matches=20)


@pytest.fixture
def weak_away() -> TeamStrength:
    return TeamStrength(name="Curacao", attack=0.55, defense_vulnerability=1.95, matches=15)


@pytest.fixture
def balanced_pair() -> tuple[TeamStrength, TeamStrength]:
    return (
        TeamStrength(name="A", attack=1.30, defense_vulnerability=1.30),
        TeamStrength(name="B", attack=1.30, defense_vulnerability=1.30),
    )


def _build_triple() -> tuple[PoissonGoalModel, BivariatePoissonModel, SkellamModel]:
    """Build Poisson + BivariatePoisson + Skellam with consistent params."""
    settings = {
        "draw_boost": 0.10,
        "draw_penalty_threshold": 0.08,
        "draw_penalty_strength": 0.05,
        "elo_gap_inflation": 0.30,
        "league_avg_multiplier": 1.18,
    }
    return (
        PoissonGoalModel(**settings),
        BivariatePoissonModel(lambda_3=0.10, **settings),
        SkellamModel(**settings),
    )


def test_probabilities_sum_to_one(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s], [0.5, 0.3, 0.2])
    pred = ens.predict(strong_home, weak_away, 1925, 1380)
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_strong_favorite_wins(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s], [0.5, 0.3, 0.2])
    pred = ens.predict(strong_home, weak_away, 1925, 1380)
    assert pred.p_home > pred.p_away
    assert pred.p_home > 0.5


def test_pure_poisson_backward_compat(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """weights=[1, 0, 0] debe dar las mismas probs que Poisson solo."""
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s], [1, 0, 0])
    ens_pred = ens.predict(strong_home, weak_away, 1925, 1380)
    p_pred = p.predict(strong_home, weak_away, 1925, 1380)
    assert math.isclose(ens_pred.p_home, p_pred.p_home, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_draw, p_pred.p_draw, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_away, p_pred.p_away, abs_tol=1e-9)


def test_pure_bivariate_poisson_backward_compat(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """weights=[0, 1, 0] debe dar las mismas probs que BP solo."""
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s], [0, 1, 0])
    ens_pred = ens.predict(strong_home, weak_away, 1925, 1380)
    bp_pred = bp.predict(strong_home, weak_away, 1925, 1380)
    assert math.isclose(ens_pred.p_home, bp_pred.p_home, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_draw, bp_pred.p_draw, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_away, bp_pred.p_away, abs_tol=1e-9)


def test_pure_skellam_backward_compat(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """weights=[0, 0, 1] debe dar las mismas probs que Skellam solo."""
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s], [0, 0, 1])
    ens_pred = ens.predict(strong_home, weak_away, 1925, 1380)
    s_pred = s.predict(strong_home, weak_away, 1925, 1380)
    assert math.isclose(ens_pred.p_home, s_pred.p_home, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_draw, s_pred.p_draw, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_away, s_pred.p_away, abs_tol=1e-9)


def test_equal_weights_average(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """weights=[1/3, 1/3, 1/3] debe ser el promedio exacto."""
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s], [1/3, 1/3, 1/3])
    ens_pred = ens.predict(strong_home, weak_away, 1925, 1380)
    p_pred = p.predict(strong_home, weak_away, 1925, 1380)
    bp_pred = bp.predict(strong_home, weak_away, 1925, 1380)
    s_pred = s.predict(strong_home, weak_away, 1925, 1380)
    expected_h = (p_pred.p_home + bp_pred.p_home + s_pred.p_home) / 3
    expected_d = (p_pred.p_draw + bp_pred.p_draw + s_pred.p_draw) / 3
    expected_a = (p_pred.p_away + bp_pred.p_away + s_pred.p_away) / 3
    assert math.isclose(ens_pred.p_home, expected_h, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_draw, expected_d, abs_tol=1e-9)
    assert math.isclose(ens_pred.p_away, expected_a, abs_tol=1e-9)


def test_default_weights_uniform() -> None:
    """Sin weights, EnsembleModel asigna pesos uniformes."""
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s])
    assert ens.weights == [1/3, 1/3, 1/3]


def test_invalid_weights_length_raises() -> None:
    p, bp, _s = _build_triple()
    with pytest.raises(ValueError, match="len\\(weights\\)"):
        EnsembleModel([p, bp], [0.5, 0.3, 0.2])


def test_invalid_weights_dont_sum_to_one_raises() -> None:
    p, bp, s = _build_triple()
    with pytest.raises(ValueError, match="sumar 1"):
        EnsembleModel([p, bp, s], [0.5, 0.5, 0.5])


def test_invalid_weights_negative_raises() -> None:
    p, bp, s = _build_triple()
    with pytest.raises(ValueError, match="negativos"):
        EnsembleModel([p, bp, s], [1.5, -0.3, -0.2])


def test_empty_models_raises() -> None:
    with pytest.raises(ValueError, match="al menos un modelo"):
        EnsembleModel([])


def test_drop_in_replacement_signature(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """Misma signature que los otros modelos."""
    p, bp, s = _build_triple()
    ens = EnsembleModel([p, bp, s], [0.5, 0.3, 0.2])
    ens_pred = ens.predict(strong_home, weak_away, 1925, 1380)
    p_pred = p.predict(strong_home, weak_away, 1925, 1380)
    assert type(ens_pred).__name__ == type(p_pred).__name__
    assert set(ens_pred.model_dump().keys()) == set(p_pred.model_dump().keys())


def test_uses_primary_most_likely_score(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """most_likely_score viene del modelo con mayor peso."""
    p, bp, s = _build_triple()
    # weights=[1, 0, 0] -> primary = Poisson
    ens = EnsembleModel([p, bp, s], [1.0, 0.0, 0.0])
    ens_pred = ens.predict(strong_home, weak_away, 1925, 1380)
    p_pred = p.predict(strong_home, weak_away, 1925, 1380)
    assert ens_pred.most_likely_score == p_pred.most_likely_score
    # weights=[0, 0, 1] -> primary = Skellam
    ens2 = EnsembleModel([p, bp, s], [0.0, 0.0, 1.0])
    ens2_pred = ens2.predict(strong_home, weak_away, 1925, 1380)
    s_pred = s.predict(strong_home, weak_away, 1925, 1380)
    assert ens2_pred.most_likely_score == s_pred.most_likely_score
