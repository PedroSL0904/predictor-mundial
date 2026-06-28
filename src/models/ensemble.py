"""Ensemble de modelos: promedio ponderado de probabilidades 1X2.

Combina predicciones de múltiples modelos (Poisson, Bivariate Poisson, Skellam)
vía promedio ponderado de probabilidades 1X2:

    p_h = Σ_i w_i * p_h_i
    p_d = Σ_i w_i * p_d_i
    p_a = Σ_i w_i * p_a_i

    con Σ w_i = 1

Backward compat:
    EnsembleModel([P, BP, S], [1, 0, 0]) == P (modelo primario con peso 1)
    EnsembleModel([P, BP, S], [0, 1, 0]) == BP
    EnsembleModel([P, BP, S], [0, 0, 1]) == S

Para el most_likely_score, usamos el modelo con mayor peso (primario).
Para lambda_home/away, idem.

Interfaz drop-in con los otros modelos: misma signature de predict().
"""
from __future__ import annotations

import numpy as np

from src.domain import MatchPrediction
from src.models.poisson import PoissonGoalModel, TeamStrength


class EnsembleModel:
    """Ensemble de modelos con pesos fijos.

    Args:
        models: lista de modelos (cualquier subclase con .predict()).
        weights: lista de pesos (debe sumar 1, mismo largo que models).

    Raises:
        ValueError: si len(models) != len(weights), o si los weights no suman 1.
    """

    def __init__(
        self,
        models: list[PoissonGoalModel | object],
        weights: list[float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("Ensemble requiere al menos un modelo")
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if len(weights) != len(models):
            raise ValueError(
                f"len(weights)={len(weights)} != len(models)={len(models)}"
            )
        total = sum(weights)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Weights deben sumar 1, suman {total}: {weights}"
            )
        if any(w < 0 for w in weights):
            raise ValueError(f"Weights no pueden ser negativos: {weights}")
        self.models = list(models)
        self.weights = list(weights)
        self.primary_idx = int(np.argmax(weights))

    def _get_primary(self) -> object:
        return self.models[self.primary_idx]

    def predict(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None = None,
        away_elo: float | None = None,
        model_name: str = "ensemble",
    ) -> MatchPrediction:
        """Promedia probabilidades 1X2 de todos los modelos según sus pesos.

        Para most_likely_score, scoreline_grid y features, usa el modelo
        primario (mayor peso).
        """
        preds = [
            m.predict(home, away, home_elo=home_elo, away_elo=away_elo)
            for m in self.models
        ]

        p_h = sum(w * p.p_home for w, p in zip(self.weights, preds, strict=True))
        p_d = sum(w * p.p_draw for w, p in zip(self.weights, preds, strict=True))
        p_a = sum(w * p.p_away for w, p in zip(self.weights, preds, strict=True))

        primary_pred = preds[self.primary_idx]

        # No contaminamos features (dict[str, float]) con metadata del ensemble.
        # El caller puede leer self.models / self.weights si los necesita.
        features = dict(primary_pred.features)

        return MatchPrediction(
            home_team=home.name,
            away_team=away.name,
            model=model_name,
            p_home=float(p_h),
            p_draw=float(p_d),
            p_away=float(p_a),
            lambda_home=primary_pred.lambda_home,
            lambda_away=primary_pred.lambda_away,
            most_likely_score=primary_pred.most_likely_score,
            most_likely_score_prob=primary_pred.most_likely_score_prob,
            scoreline_grid=primary_pred.scoreline_grid,
            features=features,
        )
