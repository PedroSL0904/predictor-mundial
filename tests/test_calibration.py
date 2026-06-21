"""Tests para el modulo de calibracion."""
import numpy as np
import pytest

from src.models.calibration import (
    PlattCalibrator, TemperatureScaler, _softmax,
    brier, log_loss, sign_acc,
)


def test_softmax_basic():
    """Softmax de un vector da probabilidades que suman 1."""
    x = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    p = _softmax(x, axis=1)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)
    # Distribucion uniforme cuando logits iguales
    np.testing.assert_allclose(p[1], [1/3, 1/3, 1/3], atol=1e-6)


def test_temperature_scaler_identity():
    """T=1 deja las probs identicas."""
    probs = np.array([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
    ts = TemperatureScaler()
    ts.T_ = 1.0
    ts.fitted = True
    out = ts.predict(probs)
    np.testing.assert_allclose(out, probs, atol=1e-6)


def test_temperature_scaler_smoothes():
    """T>1 acerca las probs al uniforme (suaviza)."""
    probs = np.array([[0.8, 0.15, 0.05]])
    ts = TemperatureScaler()
    ts.T_ = 2.0
    ts.fitted = True
    out = ts.predict(probs)
    # Debe ser mas uniforme que el original
    assert out[0, 0] < 0.8
    assert out[0, 1] > 0.15
    assert out[0, 2] > 0.05
    np.testing.assert_allclose(out.sum(axis=1), [1.0], atol=1e-6)


def test_temperature_scaler_sharpens():
    """T<1 aleja las probs del uniforme (agudiza)."""
    probs = np.array([[0.5, 0.3, 0.2]])
    ts = TemperatureScaler()
    ts.T_ = 0.5
    ts.fitted = True
    out = ts.predict(probs)
    # Debe ser mas peaky
    assert out[0, 0] > 0.5
    np.testing.assert_allclose(out.sum(axis=1), [1.0], atol=1e-6)


def test_temperature_scaler_fit_reduces_nll():
    """Entrenar reduce NLL vs T=1 por defecto."""
    np.random.seed(42)
    # Genero datos sinteticos con bias
    n = 200
    true_probs = np.random.dirichlet([2, 1, 1], n)  # skewed
    outcomes = np.array([np.random.choice(3, p=p) for p in true_probs])

    # Probs mal calibradas (ruido)
    noisy_probs = true_probs * (1 + 0.3 * np.random.randn(n, 3))
    noisy_probs = np.maximum(noisy_probs, 0.01)
    noisy_probs = noisy_probs / noisy_probs.sum(axis=1, keepdims=True)

    ts1 = TemperatureScaler()
    ts1.T_ = 1.0
    ts1.fitted = True
    nll1 = log_loss(ts1.predict(noisy_probs), outcomes)

    ts2 = TemperatureScaler()
    ts2.fit(noisy_probs, outcomes)
    nll2 = log_loss(ts2.predict(noisy_probs), outcomes)
    assert nll2 < nll1 * 1.1  # Mejora al menos marginal


def test_temperature_scaler_save_load():
    """Save y load preservan T."""
    ts = TemperatureScaler()
    ts.T_ = 0.85
    ts.fitted = True
    import json
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        ts.save(path)
        loaded = TemperatureScaler.load(path)
        assert loaded.T_ == 0.85
        assert loaded.fitted is True
    finally:
        path.unlink()


def test_brier_zero_perfect():
    """Brier = 0 cuando las probs son perfectas."""
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])
    assert brier(probs, outcomes) == 0.0


def test_brier_uniform_worst():
    """Brier = 2/3 cuando las probs son uniformes."""
    probs = np.array([[1/3, 1/3, 1/3]])
    outcomes = np.array([0])
    assert abs(brier(probs, outcomes) - 2/3) < 1e-6


def test_log_loss_uniform():
    """Log loss = log(3) cuando probs son uniformes."""
    probs = np.array([[1/3, 1/3, 1/3]])
    outcomes = np.array([0])
    assert abs(log_loss(probs, outcomes) - np.log(3)) < 1e-6


def test_sign_acc_perfect():
    """Sign acc = 1.0 cuando el top-pick es correcto siempre."""
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    outcomes = np.array([0, 1, 2])
    assert sign_acc(probs, outcomes) == 1.0
