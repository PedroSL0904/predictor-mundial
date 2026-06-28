"""Tests del backtest comparativo con/sin features historicas."""
from __future__ import annotations

import numpy as np

from src.evaluation.features_backtest import _brier, _log_loss, _rps, _sign_acc


def test_brier_perfect() -> None:
    assert _brier(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == 0.0


def test_brier_worst() -> None:
    assert _brier(np.array([[0.0, 0.0, 1.0]]), np.array([0])) == 2.0


def test_log_loss_decreases_with_confidence() -> None:
    ll_low = _log_loss(np.array([[0.5, 0.3, 0.2]]), np.array([0]))
    ll_high = _log_loss(np.array([[0.9, 0.05, 0.05]]), np.array([0]))
    assert ll_high < ll_low


def test_sign_accuracy_all_correct() -> None:
    probs = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    outs = np.array([0, 1, 2])
    assert _sign_acc(probs, outs) == 1.0


def test_rps_bounded() -> None:
    probs = np.array([[0.5, 0.3, 0.2]])
    rps = _rps(probs, np.array([0]))
    assert 0.0 <= rps <= 1.0


def test_rps_perfect_zero() -> None:
    probs = np.array([[1.0, 0.0, 0.0]])
    rps = _rps(probs, np.array([0]))
    assert abs(rps) < 1e-9
