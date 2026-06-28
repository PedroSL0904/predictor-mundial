"""Modelo Skellam para predicciones de fútbol.

La distribución Skellam modela la DIFERENCIA de dos variables Poisson
independientes:

    X = Y1 - Y2  con  Y1 ~ Poisson(λ₁),  Y2 ~ Poisson(λ₂)

Para fútbol: si Y1 = goles del local y Y2 = goles del visitante (ambos
Poisson independientes), entonces X = goles_local - goles_visitante sigue
Skellam(λ_local, λ_visitante).

PMF (forma cerrada con Bessel):
    P(X=k) = e^(-(λ₁+λ₂)) * (λ₁/λ₂)^(k/2) * I_|k|(2*sqrt(λ₁*λ₂))

donde I_v es la función de Bessel modificada de primera especie.

Para 1X2, las probabilidades se obtienen directamente:
    P(home) = Σ_{k=1}^{MAX} P(X=k)
    P(draw) = P(X=0)
    P(away) = Σ_{k=-MAX}^{-1} P(X=k)

Ventajas:
- Computacionalmente muy barato (1D en vez de 2D grid)
- Parametrización natural para el MERCADO 1X2
- scipy.stats.skellam lo implementa optimizado

Limitaciones:
- No produce joint distribution (h, a), solo el margen
- Para el marcador más probable, devolvemos una estimación basada en el modo
  de Skellam y los λ

Interfaz drop-in con PoissonGoalModel/BivariatePoissonModel: misma signature
de predict() para uso directo en el ensemble (Sprint A3).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import skellam

from src.config import get_settings
from src.domain import MatchPrediction
from src.models.poisson import TeamStrength


class SkellamModel:
    """Modelo Skellam (diferencia de goles).

    X = home_goals - away_goals ~ Skellam(λ_home, λ_away)

    λ_home, λ_away se calculan igual que en PoissonGoalModel para mantener
    consistencia con los strengths (incluye league_avg_multiplier, draw_boost,
    elo_gap_inflation).
    """

    LEAGUE_AVG_GOALS = 1.35
    ELO_GAP_THRESHOLD = 100
    MAX_MARGIN = 10  # goles de diferencia máximo (>=10 cubre cualquier caso realista)

    def __init__(
        self,
        league_avg: float = LEAGUE_AVG_GOALS,
        draw_penalty_threshold: float | None = None,
        draw_penalty_strength: float | None = None,
        elo_gap_inflation: float | None = None,
        draw_boost: float | None = 0.0,
        league_avg_multiplier: float = 1.0,
    ) -> None:
        """
        Sin `rho`, `lambda_3` ni `dispersion`: Skellam puro no tiene
        parámetros extra. Recibe los mismos ajustes de dominio (draw, elo)
        que Poisson/BP para mantener comportamiento consistente.
        """
        settings = get_settings()
        self.league_avg = league_avg
        self.league_avg_multiplier = league_avg_multiplier
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
            draw_boost if draw_boost is not None
            else settings.draw_boost
        )

    def _expected_goals(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None,
        away_elo: float | None,
    ) -> tuple[float, float]:
        """Misma fórmula que PoissonGoalModel."""
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

    def _outcome_probs(self, lam_h: float, lam_a: float) -> tuple[float, float, float]:
        """Calcula P(home), P(draw), P(away) bajo Skellam(lam_h, lam_a).

        Para P(X > 0) sumamos k=1..MAX_MARGIN (cola cerrada).
        """
        k = np.arange(-self.MAX_MARGIN, self.MAX_MARGIN + 1)
        pmf = skellam.pmf(k, lam_h, lam_a)

        p_h = float(pmf[k > 0].sum())
        p_d = float(pmf[k == 0][0])  # pmf[10] cuando MAX_MARGIN=10
        p_a = float(pmf[k < 0].sum())

        return p_h, p_d, p_a

    def _apply_draw_penalty(
        self, p_h: float, p_d: float, p_a: float
    ) -> tuple[float, float, float]:
        """Misma lógica draw_boost / draw_penalty que Poisson/BP."""
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

    def _estimate_most_likely_score(
        self, lam_h: float, lam_a: float
    ) -> tuple[tuple[int, int], float]:
        """Estima el marcador más probable.

        Skellam no da joint distribution. Usamos el modo de Skellam
        (≈ lam_h - lam_a) y los λ para construir un marcador plausible:
        - h_max = round(lam_h) (modo Poisson)
        - a_max = round(lam_a) (modo Poisson)
        - Devolvemos (h_max, a_max) que respeta el margen de Skellam.

        Para la probabilidad, usamos P(X = h_max - a_max) * normalización.
        """
        h_max = max(0, int(round(lam_h)))
        a_max = max(0, int(round(lam_a)))
        margin = h_max - a_max
        p_margin = float(skellam.pmf(margin, lam_h, lam_a))
        return (h_max, a_max), p_margin

    def predict(
        self,
        home: TeamStrength,
        away: TeamStrength,
        home_elo: float | None = None,
        away_elo: float | None = None,
        model_name: str = "skellam",
    ) -> MatchPrediction:
        """Genera predicción completa para un partido.

        Misma signature que PoissonGoalModel.predict() → drop-in replacement
        para el ensemble (Sprint A3).
        """
        lam_h, lam_a = self._expected_goals(home, away, home_elo, away_elo)
        p_h, p_d, p_a = self._outcome_probs(lam_h, lam_a)
        p_h, p_d, p_a = self._apply_draw_penalty(p_h, p_d, p_a)

        most_likely, most_likely_prob = self._estimate_most_likely_score(lam_h, lam_a)

        features = {
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "skellam_margin": lam_h - lam_a,
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
            scoreline_grid=None,  # Skellam no produce joint distribution
            features=features,
        )
