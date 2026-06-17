"""Ensemble de modelos: combina Poisson, Bivariate Poisson y xG real.

Estrategia de blending ponderado:
    p_final = w1 * p_poisson + w2 * p_bivariate + w3 * p_xg

Los pesos se pueden optimizar via grid search o usar pesos fijos.
"""
from __future__ import annotations

import numpy as np

from src.models import BivariatePoissonModel, PoissonGoalModel, TeamStrength


class EnsembleModel:
    """Ensemble que combina múltiples modelos de predicción."""

    def __init__(
        self,
        weight_poisson: float = 0.5,
        weight_bivariate: float = 0.3,
        weight_xg: float = 0.2,
        model_params: dict | None = None,
        rho_bivariate: float = 0.05,
    ):
        """
        Args:
            weight_poisson: peso del modelo Poisson base
            weight_bivariate: peso del modelo Bivariate Poisson
            weight_xg: peso del modelo con xG real (si está disponible)
            model_params: parámetros para los modelos (draw_penalty_threshold, 
                         draw_penalty_strength, elo_gap_inflation, draw_boost)
            rho_bivariate: parámetro rho para Bivariate Poisson
        """
        self.weight_poisson = weight_poisson
        self.weight_bivariate = weight_bivariate
        self.weight_xg = weight_xg

        # Normalizar pesos
        total = weight_poisson + weight_bivariate + weight_xg
        self.weight_poisson /= total
        self.weight_bivariate /= total
        self.weight_xg /= total

        # Inicializar modelos (solo parámetros del modelo, no de strengths)
        params = model_params or {}
        self.model_poisson = PoissonGoalModel(**params)
        self.model_bivariate = BivariatePoissonModel(rho=rho_bivariate, **params)
        # Para xG, usamos el mismo Poisson pero con use_xg_real=True en strengths
        self.model_xg = PoissonGoalModel(**params)

    def predict(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None = None,
        away_elo: float | None = None,
        use_xg: bool = False,
    ) -> tuple[float, float, float]:
        """Predice probabilidades combinando los modelos.

        Args:
            home: TeamStrength del equipo local
            away: TeamStrength del equipo visitante
            home_elo: Elo del equipo local
            away_elo: Elo del equipo visitante
            use_xg: si True, incluye el modelo con xG en el ensemble

        Returns:
            (p_home, p_draw, p_away) combinadas
        """
        # Predicción Poisson base
        pred_poisson = self.model_poisson.predict(
            home, away, home_elo=home_elo, away_elo=away_elo
        )
        p_poisson = np.array([pred_poisson.p_home, pred_poisson.p_draw, pred_poisson.p_away])

        # Predicción Bivariate Poisson
        pred_bivariate = self.model_bivariate.predict(
            home, away, home_elo=home_elo, away_elo=away_elo
        )
        p_bivariate = np.array([pred_bivariate.p_home, pred_bivariate.p_draw, pred_bivariate.p_away])

        # Blending
        if use_xg and self.weight_xg > 0:
            # Predicción con xG (mismo modelo pero strengths calculadas con xG real)
            pred_xg = self.model_xg.predict(
                home, away, home_elo=home_elo, away_elo=away_elo
            )
            p_xg = np.array([pred_xg.p_home, pred_xg.p_draw, pred_xg.p_away])

            p_final = (
                self.weight_poisson * p_poisson
                + self.weight_bivariate * p_bivariate
                + self.weight_xg * p_xg
            )
        else:
            # Sin xG, redistribuir peso_xg entre los otros dos
            w_p = self.weight_poisson / (self.weight_poisson + self.weight_bivariate)
            w_b = self.weight_bivariate / (self.weight_poisson + self.weight_bivariate)
            p_final = w_p * p_poisson + w_b * p_bivariate

        # Normalizar
        p_final /= p_final.sum()

        return float(p_final[0]), float(p_final[1]), float(p_final[2])
