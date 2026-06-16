"""Métricas de evaluación para modelos de predicción de fútbol.

Implementa las métricas estándar usadas en la literatura de forecasting
deportivo:
- Brier score multiclase (1X2)
- Ranked Probability Score (RPS)
- Log loss
- Accuracy de signo y de marcador exacto
"""
from __future__ import annotations

import math

import numpy as np


def brier_score(
    probs: tuple[float, float, float],
    outcome: str,
) -> float:
    """Brier score multiclase para (p_home, p_draw, p_away).

    outcome ∈ {"H", "D", "A"}
    """
    if outcome not in ("H", "D", "A"):
        raise ValueError(f"outcome debe ser H/D/A, recibí {outcome}")
    target = {"H": 0, "D": 1, "A": 2}[outcome]
    actual = np.zeros(3)
    actual[target] = 1.0
    p = np.array(probs)
    return float(np.sum((p - actual) ** 2))


def ranked_probability_score(
    probs: tuple[float, float, float],
    outcome: str,
) -> float:
    """Ranked Probability Score.

    Mide distancia entre distribuciones acumuladas. Más bajo es mejor.
    """
    if outcome not in ("H", "D", "A"):
        raise ValueError(f"outcome debe ser H/D/A, recibí {outcome}")
    target_idx = {"H": 0, "D": 1, "A": 2}[outcome]

    # CDF de las predicciones
    p = np.array(probs)
    pred_cdf = np.cumsum(p)

    # CDF del outcome real (step function)
    actual_cdf = np.zeros(3)
    actual_cdf[target_idx:] = 1.0

    return float(np.sum((pred_cdf - actual_cdf) ** 2))


def log_loss(
    probs: tuple[float, float, float],
    outcome: str,
    eps: float = 1e-15,
) -> float:
    """Log loss (cross-entropy) del outcome observado."""
    p = np.clip(np.array(probs), eps, 1.0 - eps)
    target_idx = {"H": 0, "D": 1, "A": 2}[outcome]
    return float(-math.log(p[target_idx]))


def predict_sign(
    probs: tuple[float, float, float],
) -> str:
    """Devuelve el signo (H/D/A) con mayor probabilidad."""
    p = np.array(probs)
    return ["H", "D", "A"][int(np.argmax(p))]


def sign_accuracy(
    predictions: list[tuple[float, float, float]],
    outcomes: list[str],
) -> float:
    """% de predicciones donde el signo de mayor prob coincide con el real."""
    correct = sum(
        1 for probs, out in zip(predictions, outcomes)
        if predict_sign(probs) == out
    )
    return correct / len(outcomes) if outcomes else 0.0


def exact_score_accuracy(
    predicted_scores: list[tuple[int, int]],
    actual_scores: list[tuple[int, int]],
) -> float:
    """% de marcadores exactos acertados."""
    correct = sum(1 for p, a in zip(predicted_scores, actual_scores) if p == a)
    return correct / len(actual_scores) if actual_scores else 0.0


def summarize(
    predictions: list[tuple[float, float, float]],
    outcomes: list[str],
    predicted_scores: list[tuple[int, int]] | None = None,
    actual_scores: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Resumen de métricas sobre un conjunto de predicciones."""
    if not predictions or not outcomes:
        return {}
    if len(predictions) != len(outcomes):
        raise ValueError("predictions y outcomes deben tener igual longitud")

    n = len(predictions)
    brier = sum(brier_score(p, o) for p, o in zip(predictions, outcomes)) / n
    rps = sum(ranked_probability_score(p, o) for p, o in zip(predictions, outcomes)) / n
    ll = sum(log_loss(p, o) for p, o in zip(predictions, outcomes)) / n
    sign_acc = sign_accuracy(predictions, outcomes)

    metrics = {
        "n": n,
        "brier": brier,
        "rps": rps,
        "log_loss": ll,
        "sign_accuracy": sign_acc,
        "p_home_avg": sum(p[0] for p in predictions) / n,
        "p_draw_avg": sum(p[1] for p in predictions) / n,
        "p_away_avg": sum(p[2] for p in predictions) / n,
        "outcome_freq_H": sum(1 for o in outcomes if o == "H") / n,
        "outcome_freq_D": sum(1 for o in outcomes if o == "D") / n,
        "outcome_freq_A": sum(1 for o in outcomes if o == "A") / n,
    }

    if predicted_scores is not None and actual_scores is not None:
        metrics["exact_score_accuracy"] = exact_score_accuracy(
            predicted_scores, actual_scores
        )

    return metrics
