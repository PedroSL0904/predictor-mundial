"""Bivariate Poisson: modelo que captura correlacion entre goles Home/Away.

En Poisson independiente, los goles de Home y Away son independientes.
En Bivariate Poisson, se modela:
    X = U + W
    Y = V + W
donde U, V, W son Poisson independientes con medias lambda_u, lambda_v, lambda_w.
Entonces X ~ Poisson(lambda_u + lambda_w), Y ~ Poisson(lambda_v + lambda_w),
y Cov(X, Y) = lambda_w.

Parametro rho: controla la correlacion. rho=0 es Poisson independiente.
rho > 0 captura la tendencia a que ambos equipos anoten juntos (partidos
abiertos) o no anoten (partidos cerrados).

Implementacion: usamos la formula de la PMF conjunta:
    P(X=x, Y=y) = sum_{k=0}^{min(x,y)} P(U=x-k) P(V=y-k) P(W=k)
donde U ~ Poisson(lambda_h - lambda_w), V ~ Poisson(lambda_a - lambda_w),
W ~ Poisson(lambda_w).

lambda_w = rho * sqrt(lambda_h * lambda_a)  (parametrizacion)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from src.models.poisson import PoissonGoalModel, TeamStrength, _dc_tau


class BivariatePoissonModel(PoissonGoalModel):
    """Modelo Bivariate Poisson con correlacion libre entre goles.

    Hereda de PoissonGoalModel para reusar la logica de draw_penalty,
    elo_gap_inflation, etc. Solo sobreescribe _scoreline_grid_poisson.
    """

    def __init__(self, rho: float = 0.05, **kwargs):
        """
        rho: parametro de correlacion. 0.0 = Poisson independiente.
             Tipico para futbol: 0.02-0.10.
        """
        super().__init__(**kwargs)
        self.rho_biv = rho

    def _scoreline_grid_poisson(
        self, lam_h: float, lam_a: float
    ) -> np.ndarray:
        """Grilla de probabilidades con Bivariate Poisson (vectorizado).

        Si rho_biv = 0, cae en Poisson independiente.
        """
        max_g = self.MAX_GOALS
        grid = np.zeros((max_g + 1, max_g + 1))

        if self.rho_biv <= 0:
            # Poisson independiente (vectorizado)
            goals = np.arange(max_g + 1)
            p_h = poisson.pmf(goals, lam_h)
            p_a = poisson.pmf(goals, lam_a)
            grid = np.outer(p_h, p_a)
        else:
            # Bivariate Poisson (vectorizado)
            lam_w = self.rho_biv * np.sqrt(lam_h * lam_a)
            lam_u = max(0.01, lam_h - lam_w)
            lam_v = max(0.01, lam_a - lam_w)

            goals = np.arange(max_g + 1)
            p_u = poisson.pmf(goals, lam_u)
            p_v = poisson.pmf(goals, lam_v)
            p_w = poisson.pmf(goals, lam_w)

            # Para cada (i, j), calcular sum_{k=0}^{min(i,j)} p_u[i-k] * p_v[j-k] * p_w[k]
            for i in range(max_g + 1):
                for j in range(max_g + 1):
                    max_k = min(i, j)
                    k_vals = np.arange(max_k + 1)
                    prob = np.sum(p_u[i - k_vals] * p_v[j - k_vals] * p_w[k_vals])
                    grid[i, j] = prob

        # Normalizar
        total = grid.sum()
        if total > 0:
            grid /= total
        return grid
