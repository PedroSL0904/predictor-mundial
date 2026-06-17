"""Tests para BivariatePoissonModel."""
from __future__ import annotations

import math

import pytest

from src.models import BivariatePoissonModel, TeamStrength


@pytest.fixture
def balanced_teams() -> tuple[TeamStrength, TeamStrength]:
    home = TeamStrength(name="A", attack=1.30, defense_vulnerability=1.30)
    away = TeamStrength(name="B", attack=1.30, defense_vulnerability=1.30)
    return home, away


def test_bivariate_rho_zero_equals_poisson(balanced_teams) -> None:
    """Con rho=0, Bivariate Poisson debe ser igual a Poisson independiente."""
    home, away = balanced_teams
    model_biv = BivariatePoissonModel(rho=0.0)
    model_pois = BivariatePoissonModel(rho=0.0)
    pred_biv = model_biv.predict(home, away, home_elo=1500, away_elo=1500)
    pred_pois = model_pois.predict(home, away, home_elo=1500, away_elo=1500)
    # Las probabilidades deben ser casi identicas
    assert math.isclose(pred_biv.p_home, pred_pois.p_home, abs_tol=1e-6)
    assert math.isclose(pred_biv.p_draw, pred_pois.p_draw, abs_tol=1e-6)
    assert math.isclose(pred_biv.p_away, pred_pois.p_away, abs_tol=1e-6)


def test_bivariate_rho_positive_increases_draw_probability(balanced_teams) -> None:
    """Con rho > 0, la probabilidad de empate debe aumentar (correlacion positiva)."""
    home, away = balanced_teams
    model_no_corr = BivariatePoissonModel(rho=0.0)
    model_corr = BivariatePoissonModel(rho=0.10)
    pred_no = model_no_corr.predict(home, away, home_elo=1500, away_elo=1500)
    pred_corr = model_corr.predict(home, away, home_elo=1500, away_elo=1500)
    # Con correlacion positiva, P(draw) debe aumentar
    assert pred_corr.p_draw > pred_no.p_draw


def test_bivariate_probabilities_sum_to_one(balanced_teams) -> None:
    """Las probabilidades deben sumar 1."""
    home, away = balanced_teams
    model = BivariatePoissonModel(rho=0.05)
    pred = model.predict(home, away, home_elo=1500, away_elo=1500)
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_bivariate_scoreline_grid_normalized(balanced_teams) -> None:
    """La grilla de marcadores debe sumar 1."""
    home, away = balanced_teams
    model = BivariatePoissonModel(rho=0.05)
    pred = model.predict(home, away, home_elo=1500, away_elo=1500)
    grid_sum = sum(s.probability for s in pred.scoreline_grid)
    assert math.isclose(grid_sum, 1.0, abs_tol=1e-3)


def test_bivariate_inherits_draw_boost(balanced_teams) -> None:
    """BivariatePoisson debe heredar draw_boost de PoissonGoalModel."""
    home, away = balanced_teams
    model_no_boost = BivariatePoissonModel(rho=0.05, draw_boost=0.0)
    model_boost = BivariatePoissonModel(rho=0.05, draw_boost=0.30)
    pred_no = model_no_boost.predict(home, away, home_elo=1500, away_elo=1500)
    pred_boost = model_boost.predict(home, away, home_elo=1500, away_elo=1500)
    # Con draw_boost, P(draw) debe aumentar
    assert pred_boost.p_draw > pred_no.p_draw


def test_bivariate_inherits_elo_gap_inflation(balanced_teams) -> None:
    """BivariatePoisson debe heredar elo_gap_inflation."""
    home, away = balanced_teams
    model_no_infl = BivariatePoissonModel(rho=0.05, elo_gap_inflation=0.0)
    model_infl = BivariatePoissonModel(rho=0.05, elo_gap_inflation=0.30)
    pred_no = model_no_infl.predict(home, away, home_elo=2000, away_elo=1500)
    pred_infl = model_infl.predict(home, away, home_elo=2000, away_elo=1500)
    # Con inflacion, lambda_home debe aumentar
    assert pred_infl.lambda_home > pred_no.lambda_home
