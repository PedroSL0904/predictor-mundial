"""Modelo Bivariate Poisson (Karlis & Ntzoufras, 2003) para predicciones de fútbol.

A diferencia del modelo Poisson independiente (donde los goles del local y del
visitante son independientes condicionalmente a los strengths), el Bivariate
Poisson modela explícitamente la correlación entre los goles de ambos equipos
usando tres variables latentes independientes:

    Goals_home = X + Z
    Goals_away = Y + Z

donde X, Y, Z son Poisson(λ₁, λ₂, λ₃) independientes. La variable compartida Z
captura la correlación positiva observada en fútbol: partidos con muchos goles
tienden a tener muchos goles para AMBOS equipos (no solo uno).

PMF conjunta:
    P(H=h, A=a) = e^(-(λ₁+λ₂+λ₃)) * Σ_{k=0}^{min(h,a)} λ₁^(h-k) λ₂^(a-k) λ₃^k
                                                / [(h-k)! (a-k)! k!]

Casos especiales:
    - λ₃ = 0: se reduce al modelo Poisson independiente (X, Y independientes)
    - λ₃ grande: fuerte correlación, mucha masa en la diagonal h ≈ a
    - λ₃ < 0: correlación negativa (no se usa en fútbol; Dixon-Coles lo cubre)

Interfaz drop-in con PoissonGoalModel: mismos argumentos y misma signature de
predict(), para que pueda usarse en el ensemble (Sprint A3) y como reemplazo
directo en el backtest.

Referencias:
    Karlis, D., & Ntzoufras, I. (2003). Analysis of sports data by using
    bivariate Poisson models. Journal of the Royal Statistical Society:
    Series D (The Statistician), 52(3), 381-393.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import poisson

from src.config import get_settings
from src.domain import MatchPrediction, ScorelineProb
from src.models.poisson import TeamStrength


class BivariatePoissonModel:
    """Modelo Bivariate Poisson con correlación entre goles via variable latente Z.

    λ₁ (lambda_1) = goles esperados del local sin componente compartida
    λ₂ (lambda_2) = goles esperados del visitante sin componente compartida
    λ₃ (lambda_3) = componente compartida (correlación)

    Restricciones:
        λ₁, λ₂ > 0  (siempre positivos)
        λ₃ >= 0     (en fútbol, correlación es típicamente positiva)
    """

    LEAGUE_AVG_GOALS = 1.35
    MAX_GOALS = 8
    DEFAULT_LAMBDA_3 = 0.10  # correlación típica para partidos de fútbol
    ELO_GAP_THRESHOLD = 100

    def __init__(
        self,
        league_avg: float = LEAGUE_AVG_GOALS,
        lambda_3: float = DEFAULT_LAMBDA_3,
        draw_penalty_threshold: float | None = None,
        draw_penalty_strength: float | None = None,
        elo_gap_inflation: float | None = None,
        draw_boost: float | None = 0.0,
        league_avg_multiplier: float = 1.0,
    ) -> None:
        """
        lambda_3: parámetro de correlación. 0.0 = Poisson independiente.
            Típico en fútbol: 0.05–0.20. 0.10 es un buen default.
        Resto de argumentos: igual que PoissonGoalModel para drop-in compat.
        """
        settings = get_settings()
        self.league_avg = league_avg
        self.league_avg_multiplier = league_avg_multiplier
        self.lambda_3 = max(0.0, lambda_3)
        self.draw_penalty_threshold = (
            draw_penalty_threshold
            if draw_penalty_threshold is not None
            else settings.draw_penalty_threshold
        )
        self.draw_penalty_strength = (
            draw_penalty_strength
            if draw_penalty_strength is not None
            else settings.draw_penalty_strength
        )
        self.elo_gap_inflation = (
            elo_gap_inflation
            if elo_gap_inflation is not None
            else settings.elo_gap_inflation
        )
        self.draw_boost = (
            draw_boost if draw_boost is not None and draw_boost > 0
            else settings.draw_boost
        )

    def _expected_goals(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None,
        away_elo: float | None,
    ) -> tuple[float, float]:
        """Calcula λ₁ (home) y λ₂ (away) con posible inflación por gap Elo.

        Misma fórmula que PoissonGoalModel para mantener consistencia con
        los strengths del sistema.
        """
        lam_h = self.league_avg * self.league_avg_multiplier * home.attack * away.defense_vulnerability
        lam_a = self.league_avg * self.league_avg_multiplier * away.attack * home.defense_vulnerability

        if home_elo is not None and away_elo is not None:
            elo_diff = home_elo - away_elo
            if elo_diff > self.ELO_GAP_THRESHOLD:
                factor = 1.0 + (math.log10(elo_diff / self.ELO_GAP_THRESHOLD) * self.elo_gap_inflation)
                lam_h *= factor
                lam_a /= factor
            elif elo_diff < -self.ELO_GAP_THRESHOLD:
                factor = 1.0 + (math.log10(-elo_diff / self.ELO_GAP_THRESHOLD) * self.elo_gap_inflation)
                lam_a *= factor
                lam_h /= factor

        return lam_h, lam_a

    def _scoreline_grid_bp(
        self, lam_1: float, lam_2: float, lam_3: float
    ) -> np.ndarray:
        """Grilla de probabilidades (h, a) bajo Bivariate Poisson.

        Implementación: log-sum-exp para estabilidad numérica cuando
        lam_3 * lam_1 * lam_2 son chicos y los factoriales grandes.

        Returns:
            grid shape (MAX_GOALS+1, MAX_GOALS+1) con probs que suman ≈ 1.
        """
        max_g = self.MAX_GOALS
        grid = np.zeros((max_g + 1, max_g + 1), dtype=np.float64)

        if lam_3 < 1e-12:
            # Caso degenerado: λ₃ ≈ 0 → Poisson independiente.
            # Equivalente a P(H=h, A=a) = P(X=h) * P(Y=a)
            for h in range(max_g + 1):
                for a in range(max_g + 1):
                    grid[h, a] = poisson.pmf(h, lam_1) * poisson.pmf(a, lam_2)
            return grid

        # Caso general: Σ_{k=0}^{min(h,a)} P(X=h-k) P(Y=a-k) P(Z=k)
        # donde X, Y, Z ~ Poisson(lam_1, lam_2, lam_3) independientes.
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                k_max = min(h, a)
                # log-terms para sumar estable
                log_terms = np.empty(k_max + 1, dtype=np.float64)
                for k in range(k_max + 1):
                    log_terms[k] = (
                        poisson.logpmf(h - k, lam_1)
                        + poisson.logpmf(a - k, lam_2)
                        + poisson.logpmf(k, lam_3)
                    )
                # log-sum-exp
                m = log_terms.max()
                grid[h, a] = math.exp(m) * np.exp(log_terms - m).sum()

        # Renormalizar por seguridad numérica
        total = grid.sum()
        if total > 0:
            grid /= total
        return grid

    def _outcome_probs(self, grid: np.ndarray) -> tuple[float, float, float]:
        """Suma la grilla para obtener P(Home), P(Draw), P(Away)."""
        max_g = grid.shape[0] - 1
        p_h = float(sum(
            grid[i, j]
            for i in range(max_g + 1)
            for j in range(max_g + 1)
            if i > j
        ))
        p_d = float(sum(grid[i, i] for i in range(max_g + 1)))
        p_a = float(sum(
            grid[i, j]
            for i in range(max_g + 1)
            for j in range(max_g + 1)
            if i < j
        ))
        return p_h, p_d, p_a

    def _apply_draw_penalty(
        self, p_h: float, p_d: float, p_a: float
    ) -> tuple[float, float, float]:
        """Misma lógica que PoissonGoalModel: boost en parejos, penalty en desiguales.

        Duplicado intencional (no herencia) para mantener el modelo self-contained
        y permitir divergencias futuras (ej: ajustar boost según lambda_3).
        """
        diff = abs(p_h - p_a)

        if diff <= self.draw_penalty_threshold and self.draw_boost > 0:
            proximity = 1.0 - (diff / max(self.draw_penalty_threshold, 1e-9))
            boost_factor = self.draw_boost * proximity
            avg_side = (p_h + p_a) * 0.5
            mass_to_move = avg_side * boost_factor
            p_h -= mass_to_move * 0.5
            p_a -= mass_to_move * 0.5
            p_d += mass_to_move
            total = p_h + p_d + p_a
            if total > 0:
                return p_h / total, p_d / total, p_a / total

        if diff <= self.draw_penalty_threshold:
            return p_h, p_d, p_a

        excess = min(1.0, (diff - self.draw_penalty_threshold) / 0.5)
        penalty = self.draw_penalty_strength * excess
        mass_to_redistribute = p_d * penalty
        if p_h > p_a:
            p_h += mass_to_redistribute * 0.6
            p_a += mass_to_redistribute * 0.4
        else:
            p_a += mass_to_redistribute * 0.6
            p_h += mass_to_redistribute * 0.4
        p_d *= 1.0 - penalty

        total = p_h + p_d + p_a
        return p_h / total, p_d / total, p_a / total

    def predict(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None = None,
        away_elo: float | None = None,
        model_name: str = "bivariate_poisson",
    ) -> MatchPrediction:
        """Genera predicción completa para un partido.

        Misma signature que PoissonGoalModel.predict() → drop-in replacement
        para el ensemble (Sprint A3) y el backtest (Sprint A6).
        """
        lam_1, lam_2 = self._expected_goals(home, away, home_elo, away_elo)
        grid = self._scoreline_grid_bp(lam_1, lam_2, self.lambda_3)

        p_h, p_d, p_a = self._outcome_probs(grid)
        p_h, p_d, p_a = self._apply_draw_penalty(p_h, p_d, p_a)

        idx = np.unravel_index(np.argmax(grid), grid.shape)
        most_likely = (int(idx[0]), int(idx[1]))
        most_likely_prob = float(grid[idx])

        scoreline_grid = [
            ScorelineProb(
                home_goals=i, away_goals=j, probability=float(grid[i, j])
            )
            for i in range(self.MAX_GOALS + 1)
            for j in range(self.MAX_GOALS + 1)
            if grid[i, j] > 1e-4
        ]

        features = {
            "lambda_1_home": lam_1,
            "lambda_2_away": lam_2,
            "lambda_3_shared": self.lambda_3,
            "home_attack": home.attack,
            "home_defense_vuln": home.defense_vulnerability,
            "away_attack": away.attack,
            "away_defense_vuln": away.defense_vulnerability,
        }
        if home_elo is not None and away_elo is not None:
            features["home_elo"] = home_elo
            features["away_elo"] = away_elo
            features["elo_diff"] = home_elo - away_elo

        return MatchPrediction(
            home_team=home.name,
            away_team=away.name,
            model=model_name,
            p_home=p_h,
            p_draw=p_d,
            p_away=p_a,
            lambda_home=lam_1,
            lambda_away=lam_2,
            most_likely_score=most_likely,
            most_likely_score_prob=most_likely_prob,
            scoreline_grid=scoreline_grid,
            features=features,
        )
