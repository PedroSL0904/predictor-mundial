"""Tests del reporte comparativo ensemble vs individuales."""
from __future__ import annotations

import numpy as np

from src.evaluation.ensemble_comparison import (
    _brier,
    _log_loss,
    _rps,
    _sign_acc,
    format_markdown,
)
from src.evaluation.ensemble_optimization import EnsembleWeights


def test_brier_perfect() -> None:
    assert _brier(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == 0.0


def test_log_loss_decreases_with_confidence() -> None:
    ll_low = _log_loss(np.array([[0.5, 0.3, 0.2]]), np.array([0]))
    ll_high = _log_loss(np.array([[0.9, 0.05, 0.05]]), np.array([0]))
    assert ll_high < ll_low


def test_rps_bounded() -> None:
    probs = np.array([[0.5, 0.3, 0.2]])
    rps = _rps(probs, np.array([0]))
    assert 0.0 <= rps <= 1.0


def test_rps_perfect_is_zero() -> None:
    probs = np.array([[1.0, 0.0, 0.0]])
    rps = _rps(probs, np.array([0]))
    assert abs(rps) < 1e-9


def test_sign_accuracy_all_correct() -> None:
    probs = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    outs = np.array([0, 1, 2])
    assert _sign_acc(probs, outs) == 1.0


def test_sign_accuracy_half() -> None:
    probs = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]])
    outs = np.array([0, 0])  # second prediction is wrong (predicted 1, actual 0)
    assert _sign_acc(probs, outs) == 0.5


def test_format_markdown_includes_years() -> None:
    weights = EnsembleWeights(poisson=0.5, bivariate_poisson=0.3, skellam=0.2, brier_train=0.5)
    report = {
        "rows": [
            {
                "year": 2014, "n": 64,
                "brier_ensemble": 0.55, "brier_poisson": 0.56,
                "brier_bp": 0.55, "brier_skellam": 0.55,
                "log_loss_ensemble": 1.0, "sign_acc_ensemble": 0.5,
                "sign_acc_poisson": 0.5, "elapsed": 10.0,
            },
            {
                "year": 2022, "n": 64,
                "brier_ensemble": 0.60, "brier_poisson": 0.61,
                "brier_bp": 0.60, "brier_skellam": 0.60,
                "log_loss_ensemble": 1.05, "sign_acc_ensemble": 0.45,
                "sign_acc_poisson": 0.45, "elapsed": 10.0,
            },
        ],
        "raw": {},
        "weights": weights,
    }
    md = format_markdown(report)
    assert "2014" in md
    assert "2022" in md
    assert "0.50" in md  # poisson weight
    assert "0.30" in md  # bp weight
    assert "0.20" in md  # skellam weight


def test_format_markdown_computes_averages() -> None:
    weights = EnsembleWeights(poisson=0.5, bivariate_poisson=0.3, skellam=0.2, brier_train=0.5)
    report = {
        "rows": [
            {
                "year": 2014, "n": 64,
                "brier_ensemble": 0.50, "brier_poisson": 0.50,
                "brier_bp": 0.50, "brier_skellam": 0.50,
                "log_loss_ensemble": 1.0, "sign_acc_ensemble": 0.5,
                "sign_acc_poisson": 0.5, "elapsed": 10.0,
            },
        ],
        "raw": {},
        "weights": weights,
    }
    md = format_markdown(report)
    assert "Promedios" in md
    assert "Mejora ensemble" in md
