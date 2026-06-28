"""Tests del modelo Bivariate Poisson."""
from __future__ import annotations

import math

import pytest

from src.models.bivariate_poisson import BivariatePoissonModel
from src.models.poisson import PoissonGoalModel, TeamStrength


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


def test_probabilities_sum_to_one(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    model = BivariatePoissonModel()
    pred = model.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_strong_favorite_wins(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    model = BivariatePoissonModel()
    pred = model.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    assert pred.p_home > pred.p_away
    assert pred.p_home > pred.p_draw
    assert pred.p_home > 0.5


def test_balanced_match_is_symmetric(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    home, away = balanced_pair
    model = BivariatePoissonModel()
    pred = model.predict(home, away, home_elo=1500, away_elo=1500)
    assert abs(pred.p_home - pred.p_away) < 1e-6


def test_lambda_3_increases_diagonal_mass(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    """Mayor lambda_3 debe concentrar más masa en marcadores simétricos h≈a.

    Para verificarlo: P(h=a) en la diagonal debe crecer con lambda_3.
    """
    home, away = balanced_pair
    pred_l3_low = BivariatePoissonModel(lambda_3=0.0).predict(home, away, 1500, 1500)
    pred_l3_mid = BivariatePoissonModel(lambda_3=0.10).predict(home, away, 1500, 1500)
    pred_l3_high = BivariatePoissonModel(lambda_3=0.30).predict(home, away, 1500, 1500)
    assert pred_l3_low.p_draw < pred_l3_mid.p_draw < pred_l3_high.p_draw


def test_lambda_3_zero_reduces_to_independent_poisson(
    balanced_pair: tuple[TeamStrength, TeamStrength],
) -> None:
    """Con lambda_3=0, Bivariate Poisson debe coincidir con Poisson independiente
    puro (sin Dixon-Coles). PoissonGoalModel siempre aplica DC, por lo que
    la comparación es con el producto de dos Poisson PDF directas.

    Para lambda ~ 2.28, P(0-0) bajo Poisson puro:
        P(X=0) = e^-2.28 ≈ 0.1023
        P(0-0) = 0.1023² ≈ 0.01047

    Mientras DC(ρ=-0.03) lo ajusta a ~0.0121.
    """
    from scipy.stats import poisson as _pois

    home, away = balanced_pair
    bp = BivariatePoissonModel(lambda_3=0.0)
    bp_pred = bp.predict(home, away, 1500, 1500)

    # Calcular P(0-0) y P(1-1) bajo Poisson independiente puro
    lam = bp_pred.lambda_home
    p0 = _pois.pmf(0, lam)
    p1 = _pois.pmf(1, lam)
    expected_00 = p0 * p0
    expected_11 = p1 * p1

    # Buscar en scoreline_grid del BP
    grid = {(s.home_goals, s.away_goals): s.probability for s in bp_pred.scoreline_grid}
    assert math.isclose(grid[(0, 0)], expected_00, abs_tol=1e-4)
    assert math.isclose(grid[(1, 1)], expected_11, abs_tol=1e-4)


def test_lambda_3_increases_p_draw(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    """Independientemente del draw_boost, lambda_3 aumenta P(draw) en parejos."""
    home, away = balanced_pair
    p_indep = BivariatePoissonModel(lambda_3=0.0, draw_boost=0.0).predict(home, away, 1500, 1500)
    p_corr = BivariatePoissonModel(lambda_3=0.20, draw_boost=0.0).predict(home, away, 1500, 1500)
    assert p_corr.p_draw > p_indep.p_draw


def test_drop_in_replacement_signature(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """Misma signature que PoissonGoalModel.predict() — drop-in replacement."""
    bp = BivariatePoissonModel()
    pois = PoissonGoalModel()
    bp_pred = bp.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    pois_pred = pois.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    # Tipo de retorno
    assert type(bp_pred).__name__ == type(pois_pred).__name__
    # Mismos campos
    assert set(bp_pred.model_dump().keys()) == set(pois_pred.model_dump().keys())
    # model name diferente
    assert bp_pred.model == "bivariate_poisson"
    assert pois_pred.model == "poisson_dc_xg"


def test_grid_sums_to_one(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    """La grilla completa de scoreline_grid debe sumar ≈ 1 (truncada a 1e-4)."""
    home, away = balanced_pair
    pred = BivariatePoissonModel().predict(home, away, 1500, 1500)
    assert pred.scoreline_grid is not None
    total = sum(s.probability for s in pred.scoreline_grid)
    # La grilla solo incluye marcadores con prob > 1e-4 (truncado).
    # Debe ser >= 1 - max_goals_truncation.
    # Para lambda ~ 1.3, P(>8) es ~0, así que la suma debe estar cerca de 1.
    assert total > 0.99


def test_elo_inflation_applies(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    """Si hay gap Elo grande, lambda_1 del favorito debe inflarse."""
    home, away = balanced_pair
    pred_no_elo = BivariatePoissonModel().predict(home, away)
    pred_with_elo = BivariatePoissonModel().predict(home, away, 1900, 1400)
    assert pred_with_elo.lambda_home > pred_no_elo.lambda_home
    assert pred_with_elo.lambda_away < pred_no_elo.lambda_away


def test_league_avg_multiplier_inflates_lambdas(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """league_avg_multiplier escala ambas lambdas proporcionalmente."""
    bp_1 = BivariatePoissonModel(league_avg_multiplier=1.0)
    bp_2 = BivariatePoissonModel(league_avg_multiplier=1.18)
    p1 = bp_1.predict(strong_home, weak_away, 1500, 1500)
    p2 = bp_2.predict(strong_home, weak_away, 1500, 1500)
    ratio = p2.lambda_home / p1.lambda_home
    assert math.isclose(ratio, 1.18, abs_tol=1e-6)


def test_invalid_lambda_3_negative_clamps_to_zero() -> None:
    """lambda_3 negativo se clampa a 0 (correlación negativa se modela via DC)."""
    model = BivariatePoissonModel(lambda_3=-0.5)
    assert model.lambda_3 == 0.0
