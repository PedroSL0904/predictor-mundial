"""Tests del modulo de optimizacion del ensemble."""
from __future__ import annotations

import numpy as np

from src.evaluation.ensemble_optimization import (
    EnsembleWeights,
    _brier,
    optimize_weights_for_set,
)


def test_brier_perfect_prediction() -> None:
    probs = np.array([[1.0, 0.0, 0.0]])
    outcomes = np.array([0])
    assert _brier(probs, outcomes) == 0.0


def test_brier_worst_case() -> None:
    probs = np.array([[0.0, 0.0, 1.0]])
    outcomes = np.array([0])
    assert _brier(probs, outcomes) == 2.0


def test_optimize_returns_simplex() -> None:
    """Los pesos optimizados deben sumar 1 y ser >= 0."""
    np.random.seed(42)
    n = 50
    probs_p = np.random.dirichlet([1, 1, 1], size=n)
    probs_bp = np.random.dirichlet([1, 1, 1], size=n)
    probs_sk = np.random.dirichlet([1, 1, 1], size=n)
    outcomes = np.random.randint(0, 3, size=n)
    (w1, w2, w3), brier = optimize_weights_for_set(probs_p, probs_bp, probs_sk, outcomes)
    assert abs(w1 + w2 + w3 - 1.0) < 1e-6
    assert w1 >= 0 and w2 >= 0 and w3 >= 0
    assert brier < 2.0


def test_optimize_pure_poisson_when_models_agree() -> None:
    """Si los 3 modelos dan las mismas probs, el optimo es cualquier simplex
    point. El brier debe coincidir con el de Poisson solo."""
    np.random.seed(42)
    n = 50
    probs = np.random.dirichlet([2, 2, 2], size=n)
    outcomes = np.random.randint(0, 3, size=n)
    (w1, w2, w3), brier_opt = optimize_weights_for_set(probs, probs, probs, outcomes)
    brier_p = _brier(probs, outcomes)
    # Como todos los modelos coinciden, cualquier combinacion da el mismo brier
    assert abs(brier_opt - brier_p) < 1e-6


def test_optimize_chooses_better_model() -> None:
    """Si Poisson es perfecto y los otros son ruido, el optimo debe elegir
    weights cercanos a [1, 0, 0]."""
    np.random.seed(42)
    n = 99  # multiplo de 3 para que [0,1,2]*33 funcione
    # Poisson "perfecto": probs muy concentrated en el outcome correcto
    outcomes_cycle = [0, 1, 2] * (n // 3)
    probs_p = np.array([
        [0.9, 0.05, 0.05] if o == 0 else
        [0.05, 0.9, 0.05] if o == 1 else
        [0.05, 0.05, 0.9]
        for o in outcomes_cycle
    ])
    # BP y Skellam: ruido
    probs_bp = np.random.dirichlet([1, 1, 1], size=n)
    probs_sk = np.random.dirichlet([1, 1, 1], size=n)
    outcomes = np.array(outcomes_cycle)
    (w1, w2, w3), brier_opt = optimize_weights_for_set(probs_p, probs_bp, probs_sk, outcomes)
    # El optimo debe preferir Poisson
    assert w1 >= 0.5


def test_optimize_handles_empty_set() -> None:
    """Sin datos, devuelve (1, 0, 0) y brier=inf."""
    probs = np.empty((0, 3))
    outcomes = np.empty((0,), dtype=np.int64)
    (w1, w2, w3), brier = optimize_weights_for_set(probs, probs, probs, outcomes)
    assert (w1, w2, w3) == (1.0, 0.0, 0.0)
    assert brier == float("inf")


def test_ensemble_weights_str() -> None:
    w = EnsembleWeights(poisson=0.5, bivariate_poisson=0.3, skellam=0.2, brier_train=0.55)
    s = str(w)
    assert "0.50" in s
    assert "0.30" in s
    assert "0.20" in s
    assert "0.5500" in s


def test_ensemble_weights_sum_to_one() -> None:
    w = EnsembleWeights(poisson=0.4, bivariate_poisson=0.35, skellam=0.25, brier_train=0.5)
    total = sum(w.as_list())
    assert abs(total - 1.0) < 1e-9
