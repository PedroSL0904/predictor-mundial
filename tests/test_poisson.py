"""Tests del modelo Poisson y métricas de evaluación."""
from __future__ import annotations

import math

import pytest

from src.evaluation import (
    brier_score,
    log_loss,
    ranked_probability_score,
    sign_accuracy,
    summarize,
)
from src.models import PoissonGoalModel, TeamStrength


@pytest.fixture
def strong_home() -> TeamStrength:
    return TeamStrength(
        name="Germany", attack=2.10, defense_vulnerability=0.85, matches=20
    )


@pytest.fixture
def weak_away() -> TeamStrength:
    return TeamStrength(
        name="Curacao", attack=0.55, defense_vulnerability=1.95, matches=15
    )


def test_model_predicts_home_win_when_stronger(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    model = PoissonGoalModel()
    pred = model.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    assert pred.p_home > pred.p_away
    assert pred.p_home > pred.p_draw
    assert pred.p_home > 0.5


def test_probabilities_sum_to_one(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    model = PoissonGoalModel()
    pred = model.predict(strong_home, weak_away)
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_lambda_scales_with_strength(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    model = PoissonGoalModel()
    pred = model.predict(strong_home, weak_away)
    assert pred.lambda_home > pred.lambda_away


def test_anti_draw_penalty_reduces_p_draw_for_clear_favorite(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    # Sin penalización: P(draw) sería ~10-15%
    # Con penalización: P(draw) debe ser claramente menor
    model = PoissonGoalModel()
    pred = model.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    assert pred.p_draw < 0.15  # debería ser muy bajo


def test_draw_penalty_does_not_apply_to_balanced_match() -> None:
    home = TeamStrength(name="A", attack=1.30, defense_vulnerability=1.30)
    away = TeamStrength(name="B", attack=1.30, defense_vulnerability=1.30)
    model = PoissonGoalModel()
    pred = model.predict(home, away, home_elo=1500, away_elo=1500)
    # Sin gap, no debería haber penalización: P(draw) > 0.18 (Poisson puro ~0.20)
    # y p_home == p_away (partido perfectamente simétrico)
    assert pred.p_draw > 0.18
    assert abs(pred.p_home - pred.p_away) < 1e-6


def test_strong_favorite_has_low_p_draw() -> None:
    # Verifica que el sesgo anti-1-1 funciona: favorito claro → draw muy bajo
    home = TeamStrength(name="Top", attack=2.0, defense_vulnerability=0.9)
    away = TeamStrength(name="Bottom", attack=0.6, defense_vulnerability=2.0)
    model = PoissonGoalModel()
    pred = model.predict(home, away, home_elo=2000, away_elo=1300)
    assert pred.p_draw < 0.12
    assert pred.p_home > 0.70


def test_draw_boost_increases_p_draw_for_balanced_match() -> None:
    # En un partido parejo, draw_boost debe subir P(draw) por encima del Poisson puro
    home = TeamStrength(name="A", attack=1.30, defense_vulnerability=1.30)
    away = TeamStrength(name="B", attack=1.30, defense_vulnerability=1.30)
    model_no = PoissonGoalModel(draw_boost=0.0)
    model_yes = PoissonGoalModel(draw_boost=0.50)
    pred_no = model_no.predict(home, away, home_elo=1500, away_elo=1500)
    pred_yes = model_yes.predict(home, away, home_elo=1500, away_elo=1500)
    assert pred_yes.p_draw > pred_no.p_draw
    # Suma sigue siendo ~1
    total = pred_yes.p_home + pred_yes.p_draw + pred_yes.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_draw_boost_does_not_affect_heavy_favorite() -> None:
    # En partido muy desigual, draw_boost no debe aplicar (gap > threshold)
    home = TeamStrength(name="Top", attack=2.0, defense_vulnerability=0.9)
    away = TeamStrength(name="Bottom", attack=0.6, defense_vulnerability=2.0)
    model_no = PoissonGoalModel(draw_boost=0.0)
    model_yes = PoissonGoalModel(draw_boost=0.50)
    pred_no = model_no.predict(home, away, home_elo=2000, away_elo=1300)
    pred_yes = model_yes.predict(home, away, home_elo=2000, away_elo=1300)
    # En partido desigual, draw_boost no debería cambiar (gap supera threshold)
    assert math.isclose(pred_no.p_draw, pred_yes.p_draw, abs_tol=1e-4)


def test_dispersion_increases_variance_of_grid() -> None:
    # Negative Binomial debe tener varianza > Poisson para mismos lambda
    home = TeamStrength(name="A", attack=1.5, defense_vulnerability=1.2)
    away = TeamStrength(name="B", attack=1.2, defense_vulnerability=1.3)
    m_pois = PoissonGoalModel(dispersion=0.0)
    m_nb = PoissonGoalModel(dispersion=0.20)
    p_pois = m_pois.predict(home, away)
    p_nb = m_nb.predict(home, away)
    # Las lambdas son identicas
    assert math.isclose(p_pois.lambda_home, p_nb.lambda_home, abs_tol=1e-9)
    # Pero la masa en marcadores extremos (>3 goles) es mayor con NB
    # P(>=4) total = 1 - P(<=3)
    def p_ge4(grid):
        if grid is None:
            return 0.0
        return sum(s.probability for s in grid if s.home_goals >= 4 or s.away_goals >= 4)
    assert p_ge4(p_nb.scoreline_grid) > p_ge4(p_pois.scoreline_grid)


def test_brier_score_perfect_prediction() -> None:
    assert brier_score((1.0, 0.0, 0.0), "H") == 0.0


def test_brier_score_worst_case() -> None:
    # Pronóstico muy errado
    score = brier_score((0.0, 0.0, 1.0), "H")
    assert score == pytest.approx(2.0, abs=1e-6)


def test_log_loss_decreases_with_confidence() -> None:
    # Más confianza → menos log loss
    ll_low = log_loss((0.5, 0.3, 0.2), "H")
    ll_high = log_loss((0.9, 0.05, 0.05), "H")
    assert ll_high < ll_low


def test_rps_bounded() -> None:
    rps = ranked_probability_score((0.5, 0.3, 0.2), "H")
    assert 0.0 <= rps <= 1.0


def test_sign_accuracy() -> None:
    preds = [(0.6, 0.2, 0.2), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5)]
    outcomes = ["H", "D", "A"]
    assert sign_accuracy(preds, outcomes) == 1.0


def test_summarize_returns_all_metrics() -> None:
    preds = [(0.6, 0.2, 0.2), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5)]
    outcomes = ["H", "D", "A"]
    metrics = summarize(preds, outcomes)
    assert "brier" in metrics
    assert "rps" in metrics
    assert "log_loss" in metrics
    assert "sign_accuracy" in metrics
    assert metrics["n"] == 3


def test_elo_inflation_increases_favorite_lambda() -> None:
    # Con gap Elo grande, la inflación debería aumentar la λ del favorito
    home = TeamStrength(name="Strong", attack=1.5, defense_vulnerability=1.0)
    away = TeamStrength(name="Weak", attack=0.8, defense_vulnerability=1.5)

    model_no_elo = PoissonGoalModel()
    pred_no_elo = model_no_elo.predict(home, away)

    model_with_elo = PoissonGoalModel()
    pred_with_elo = model_with_elo.predict(home, away, home_elo=1900, away_elo=1400)

    assert pred_with_elo.lambda_home > pred_no_elo.lambda_home
    assert pred_with_elo.lambda_away < pred_no_elo.lambda_away
