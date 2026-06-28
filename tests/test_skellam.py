"""Tests del modelo Skellam."""
from __future__ import annotations

import math

import pytest
from scipy.stats import skellam

from src.models.poisson import TeamStrength
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


def test_probabilities_sum_to_one(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    model = SkellamModel()
    pred = model.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    total = pred.p_home + pred.p_draw + pred.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_strong_favorite_wins(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    model = SkellamModel()
    pred = model.predict(strong_home, weak_away, home_elo=1925, away_elo=1380)
    assert pred.p_home > pred.p_away
    assert pred.p_home > pred.p_draw
    assert pred.p_home > 0.5


def test_balanced_match_is_symmetric(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    home, away = balanced_pair
    model = SkellamModel()
    pred = model.predict(home, away, home_elo=1500, away_elo=1500)
    assert abs(pred.p_home - pred.p_away) < 1e-6


def test_p_draw_matches_skellam_pmf_at_zero(
    balanced_pair: tuple[TeamStrength, TeamStrength],
) -> None:
    """P(draw) = P(X=0) donde X ~ Skellam(λ_h, λ_a)."""
    home, away = balanced_pair
    pred = SkellamModel(draw_boost=0.0, draw_penalty_strength=0.0).predict(
        home, away, 1500, 1500
    )
    expected_p_draw = float(skellam.pmf(0, pred.lambda_home, pred.lambda_away))
    assert math.isclose(pred.p_draw, expected_p_draw, abs_tol=1e-6)


def test_p_home_equals_sum_of_positive_margin(
    balanced_pair: tuple[TeamStrength, TeamStrength],
) -> None:
    """P(home) = Σ_{k=1}^{MAX} P(X=k)."""
    home, away = balanced_pair
    pred = SkellamModel(draw_boost=0.0, draw_penalty_strength=0.0).predict(
        home, away, 1500, 1500
    )
    k = list(range(1, 11))
    expected = sum(
        skellam.pmf(ki, pred.lambda_home, pred.lambda_away) for ki in k
    )
    assert math.isclose(pred.p_home, expected, abs_tol=1e-6)


def test_p_away_equals_sum_of_negative_margin(
    balanced_pair: tuple[TeamStrength, TeamStrength],
) -> None:
    """P(away) = Σ_{k=-MAX}^{-1} P(X=k)."""
    home, away = balanced_pair
    pred = SkellamModel(draw_boost=0.0, draw_penalty_strength=0.0).predict(
        home, away, 1500, 1500
    )
    k = list(range(-10, 0))
    expected = sum(
        skellam.pmf(ki, pred.lambda_home, pred.lambda_away) for ki in k
    )
    assert math.isclose(pred.p_away, expected, abs_tol=1e-6)


def test_most_likely_score_uses_rounded_lambdas(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """most_likely_score ≈ (round(λ_h), round(λ_a))."""
    model = SkellamModel()
    pred = model.predict(strong_home, weak_away, 1500, 1500)
    expected_h = max(0, int(round(pred.lambda_home)))
    expected_a = max(0, int(round(pred.lambda_away)))
    assert pred.most_likely_score == (expected_h, expected_a)


def test_drop_in_replacement_signature(
    strong_home: TeamStrength, weak_away: TeamStrength
) -> None:
    """Misma signature que PoissonGoalModel/BivariatePoissonModel."""
    from src.models.bivariate_poisson import BivariatePoissonModel
    from src.models.poisson import PoissonGoalModel

    sk = SkellamModel()
    bp = BivariatePoissonModel()
    pois = PoissonGoalModel()
    sk_pred = sk.predict(strong_home, weak_away, 1925, 1380)
    bp_pred = bp.predict(strong_home, weak_away, 1925, 1380)
    pois_pred = pois.predict(strong_home, weak_away, 1925, 1380)
    # Mismo tipo
    assert type(sk_pred).__name__ == type(bp_pred).__name__ == type(pois_pred).__name__
    # Mismos campos top-level
    sk_keys = set(sk_pred.model_dump().keys())
    assert sk_keys == set(bp_pred.model_dump().keys())
    # model name diferente
    assert sk_pred.model == "skellam"


def test_scoreline_grid_is_none(strong_home: TeamStrength, weak_away: TeamStrength) -> None:
    """Skellam no produce joint distribution, scoreline_grid debe ser None."""
    model = SkellamModel()
    pred = model.predict(strong_home, weak_away)
    assert pred.scoreline_grid is None


def test_elo_inflation_applies(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    home, away = balanced_pair
    pred_no_elo = SkellamModel().predict(home, away)
    pred_with_elo = SkellamModel().predict(home, away, 1900, 1400)
    assert pred_with_elo.lambda_home > pred_no_elo.lambda_home
    assert pred_with_elo.lambda_away < pred_no_elo.lambda_away


def test_draw_boost_increases_p_draw(balanced_pair: tuple[TeamStrength, TeamStrength]) -> None:
    home, away = balanced_pair
    p_no = SkellamModel(draw_boost=0.0).predict(home, away, 1500, 1500)
    p_yes = SkellamModel(draw_boost=0.50).predict(home, away, 1500, 1500)
    assert p_yes.p_draw > p_no.p_draw
    # Suma sigue siendo 1
    total = p_yes.p_home + p_yes.p_draw + p_yes.p_away
    assert math.isclose(total, 1.0, abs_tol=1e-6)
