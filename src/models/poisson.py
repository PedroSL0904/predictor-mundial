"""Motor de predicción de partidos de fútbol.

Implementa un modelo Poisson bivariado con ajuste Dixon-Coles, alimentado
por xG histórico. Incluye correcciones para los dos sesgos principales
observados en Oloraculo:

1. **Anti-sesgo 1-1**: cuando |p_home - p_away| > umbral, penaliza la
   probabilidad de empate (los empates están sistemáticamente sobreestimados
   en modelos Poisson puros).
2. **Inflación de λ por gap Elo**: cuando la diferencia de fuerza es grande
   (Curacao vs Germany), escala la λ del favorito para reflejar mejor las
   goleadas observadas.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from scipy.stats import nbinom, poisson
from pydantic import BaseModel, Field

from src.config import get_settings
from src.domain import MatchOutcome, MatchPrediction, ScorelineProb


# Dixon-Coles correction: tau(x, y, lam_h, lam_a, rho)
def _dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class TeamStrength(BaseModel):
    """Fuerza relativa de un equipo (calculada a partir de xG)."""

    name: str
    attack: float = Field(gt=0.0, description="xG promedio a favor por partido")
    defense_vulnerability: float = Field(
        gt=0.0, description="xG promedio en contra por partido"
    )
    matches: int = 0
    last_updated: datetime | None = None


class PoissonGoalModel:
    """Modelo Poisson bivariado con Dixon-Coles y correcciones anti-sesgo.

    λ_home = league_avg * attack_home * defense_vulnerability_away
    λ_away = league_avg * attack_away * defense_vulnerability_home

    El ajuste de Dixon-Coles corrige la correlación negativa entre los
    resultados de bajo marcador (0-0, 1-0, 0-1, 1-1).
    """

    LEAGUE_AVG_GOALS = 1.35  # promedio goles por equipo en partidos internacionales recientes
    MAX_GOALS = 8
    DEFAULT_RHO = -0.03  # correlación Dixon-Coles

    def __init__(
        self,
        league_avg: float = LEAGUE_AVG_GOALS,
        rho: float = DEFAULT_RHO,
        draw_penalty_threshold: float | None = None,
        draw_penalty_strength: float | None = None,
        elo_gap_inflation: float | None = None,
    ) -> None:
        settings = get_settings()
        self.league_avg = league_avg
        self.rho = rho
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

    def _expected_goals(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None = None,
        away_elo: float | None = None,
    ) -> tuple[float, float]:
        """Calcula λ_home y λ_away con posible inflación por gap Elo."""
        lam_h = self.league_avg * home.attack * away.defense_vulnerability
        lam_a = self.league_avg * away.attack * home.defense_vulnerability

        # Inflación por gap Elo: si hay ratings, escala λ del favorito
        if home_elo is not None and away_elo is not None:
            elo_diff = home_elo - away_elo
            # elo_diff positivo = local más fuerte
            if elo_diff > 100:
                # Local claramente más fuerte
                factor = 1.0 + (math.log10(elo_diff / 100) * self.elo_gap_inflation)
                lam_h *= factor
                lam_a /= factor
            elif elo_diff < -100:
                # Visitante claramente más fuerte
                factor = 1.0 + (math.log10(-elo_diff / 100) * self.elo_gap_inflation)
                lam_a *= factor
                lam_h /= factor

        return lam_h, lam_a

    def _scoreline_grid_poisson(
        self, lam_h: float, lam_a: float
    ) -> np.ndarray:
        """Grilla de probabilidades marcador×marcador (Poisson independiente)."""
        max_g = self.MAX_GOALS
        grid = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                grid[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        return grid

    def _scoreline_grid_dc(
        self, lam_h: float, lam_a: float
    ) -> np.ndarray:
        """Grilla con corrección Dixon-Coles."""
        base = self._scoreline_grid_poisson(lam_h, lam_a)
        max_g = self.MAX_GOALS
        for i in range(min(2, max_g + 1)):
            for j in range(min(2, max_g + 1)):
                base[i, j] *= _dc_tau(i, j, lam_h, lam_a, self.rho)
        # Renormalizar
        total = base.sum()
        if total > 0:
            base /= total
        return base

    def _outcome_probs(self, grid: np.ndarray) -> tuple[float, float, float]:
        """Suma grilla para obtener P(Home), P(Draw), P(Away)."""
        max_g = grid.shape[0] - 1
        p_h = sum(
            grid[i, j]
            for i in range(max_g + 1)
            for j in range(max_g + 1)
            if i > j
        )
        p_d = sum(grid[i, i] for i in range(max_g + 1))
        p_a = sum(
            grid[i, j]
            for i in range(max_g + 1)
            for j in range(max_g + 1)
            if i < j
        )
        return p_h, p_d, p_a

    def _apply_draw_penalty(
        self, p_h: float, p_d: float, p_a: float
    ) -> tuple[float, float, float]:
        """Reduce P(draw) cuando hay asimetría clara entre local y visitante.

        El modelo Poisson sobreestima empates porque trata Home/Draw/Away como
        simétricos. En la práctica, cuando un equipo es claramente favorito,
        los empates son menos probables de lo que sugiere la grilla.
        """
        diff = abs(p_h - p_a)
        if diff <= self.draw_penalty_threshold:
            return p_h, p_d, p_a

        # Penalización proporcional al gap
        excess = min(1.0, (diff - self.draw_penalty_threshold) / 0.5)
        penalty = self.draw_penalty_strength * excess
        mass_to_redistribute = p_d * penalty

        # Redistribuir 60/40 hacia el favorito
        if p_h > p_a:
            p_h += mass_to_redistribute * 0.6
            p_a += mass_to_redistribute * 0.4
        else:
            p_a += mass_to_redistribute * 0.6
            p_h += mass_to_redistribute * 0.4
        p_d *= 1.0 - penalty

        # Renormalizar
        total = p_h + p_d + p_a
        return p_h / total, p_d / total, p_a / total

    def predict(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None = None,
        away_elo: float | None = None,
        model_name: str = "poisson_dc_xg",
    ) -> MatchPrediction:
        """Genera predicción completa para un partido."""
        lam_h, lam_a = self._expected_goals(home, away, home_elo, away_elo)
        grid = self._scoreline_grid_dc(lam_h, lam_a)

        p_h, p_d, p_a = self._outcome_probs(grid)
        p_h, p_d, p_a = self._apply_draw_penalty(p_h, p_d, p_a)

        # Marcador más probable
        idx = np.unravel_index(np.argmax(grid), grid.shape)
        most_likely = (int(idx[0]), int(idx[1]))
        most_likely_prob = float(grid[idx])

        # Grilla completa para el output
        scoreline_grid = [
            ScorelineProb(
                home_goals=i, away_goals=j, probability=float(grid[i, j])
            )
            for i in range(self.MAX_GOALS + 1)
            for j in range(self.MAX_GOALS + 1)
            if grid[i, j] > 1e-4
        ]

        features = {
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "rho": self.rho,
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
            lambda_home=lam_h,
            lambda_away=lam_a,
            most_likely_score=most_likely,
            most_likely_score_prob=most_likely_prob,
            scoreline_grid=scoreline_grid,
            features=features,
        )

    def predict_score(self, lam_h: float, lam_a: float) -> tuple[int, int]:
        """Muestrea un marcador de la distribución."""
        return int(poisson.rvs(lam_h)), int(poisson.rvs(lam_a))
