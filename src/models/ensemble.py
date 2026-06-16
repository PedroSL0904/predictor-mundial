"""Ensemble que combina Poisson del modelo con mercado.

La idea: las cuotas de Pinnacle son el mejor "baseline" disponible
(consensúan miles de apostadores). Un ensemble ponderado entre modelo
y mercado suele bajar Brier ~0.04-0.06 respecto a cada uno por separado.

peso_modelo = 1 - peso_mercado

Estrategias implementadas:
- Fixed: peso constante (default 0.4 modelo, 0.6 mercado)
- Confidence-weighted: cuando el modelo está muy seguro, le da más peso
- Disagreement-penalized: cuando modelo y mercado discrepan mucho, cae al mercado
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from src.domain import MatchOutcome, MatchPrediction


@dataclass
class MarketOdds:
    """Probabilidades implícitas de mercado (ya normalizadas, sin vig)."""
    p_home: float
    p_draw: float
    p_away: float

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.p_home, self.p_draw, self.p_away)


def ensemble_fixed(
    model: MatchPrediction,
    market: MarketOdds,
    model_weight: float = 0.4,
) -> tuple[float, float, float]:
    """Mezcla lineal fija: alpha * modelo + (1-alpha) * mercado."""
    a = model_weight
    p_h = a * model.p_home + (1 - a) * market.p_home
    p_d = a * model.p_draw + (1 - a) * market.p_draw
    p_a = a * model.p_away + (1 - a) * market.p_away
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


def ensemble_disagreement(
    model: MatchPrediction,
    market: MarketOdds,
    base_weight: float = 0.4,
) -> tuple[float, float, float]:
    """Cuando modelo y mercado difieren mucho, confia más en el mercado.

    Mide la "distancia" Jensen-Shannon entre las dos distribuciones.
    Si JS > 0.1, baja el peso del modelo a 0.2.
    """
    p_m = [model.p_home, model.p_draw, model.p_away]
    p_k = [market.p_home, market.p_draw, market.p_away]

    js = _jensen_shannon(p_m, p_k)

    if js > 0.10:
        weight = 0.20
    elif js > 0.05:
        weight = base_weight * 0.6
    else:
        weight = base_weight

    return ensemble_fixed(model, market, model_weight=weight)


def ensemble_confidence_weighted(
    model: MatchPrediction,
    market: MarketOdds,
    base_weight: float = 0.4,
) -> tuple[float, float, float]:
    """Cuando el modelo está muy seguro, le da más peso.

    Mide la "entropía" del modelo (menor = más seguro).
    """
    entropy = _entropy([model.p_home, model.p_draw, model.p_away])
    max_entropy = log(3)  # entropía máxima con 3 clases uniformes
    confidence = 1.0 - (entropy / max_entropy)  # 0 (inseguro) a 1 (muy seguro)

    # Escalar peso: 0.2 (inseguro) a 0.6 (muy seguro)
    weight = 0.2 + (0.4 * confidence)
    # Pero no pasar del base_weight
    weight = min(weight, base_weight * 1.5)

    return ensemble_fixed(model, market, model_weight=weight)


def _entropy(probs: list[float]) -> float:
    """Entropía de Shannon en nats."""
    return -sum(p * log(p) for p in probs if p > 0)


def _jensen_shannon(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence (raíz cuadrada del JS)."""
    import math

    def kl(a, b):
        return sum(ai * log(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)

    m = [(a + b) / 2 for a, b in zip(p, q)]
    return math.sqrt((kl(p, m) + kl(q, m)) / 2)
