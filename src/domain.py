"""Modelos de dominio del sistema de predicciones."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MatchOutcome(str, Enum):
    """Resultado de un partido en términos de 1X2."""

    HOME = "H"
    DRAW = "D"
    AWAY = "A"


class MatchStage(str, Enum):
    """Etapa del torneo."""

    GROUP = "group"
    ROUND_32 = "R32"
    ROUND_16 = "R16"
    QUARTER = "QF"
    SEMI = "SF"
    THIRD = "3rd"
    FINAL = "F"


class Team(BaseModel):
    """Representación de una selección."""

    name: str
    code: str | None = None
    fifa_points: float | None = None
    elo: float | None = None
    confederation: str | None = None


class MatchResult(BaseModel):
    """Resultado real de un partido."""

    home_team: str
    away_team: str
    date: datetime
    home_goals: int
    away_goals: int
    neutral_venue: bool = True
    tournament: str | None = None
    home_xg: float | None = None
    away_xg: float | None = None


class ScorelineProb(BaseModel):
    """Probabilidad de un marcador exacto (i, j)."""

    home_goals: int
    away_goals: int
    probability: float = Field(ge=0.0, le=1.0)


class MatchPrediction(BaseModel):
    """Predicción completa de un partido."""

    home_team: str
    away_team: str
    model: str

    # Probabilidades 1X2
    p_home: float = Field(ge=0.0, le=1.0)
    p_draw: float = Field(ge=0.0, le=1.0)
    p_away: float = Field(ge=0.0, le=1.0)

    # Goles esperados
    lambda_home: float
    lambda_away: float

    # Marcador más probable
    most_likely_score: tuple[int, int]
    most_likely_score_prob: float

    # Grilla completa de marcadores (opcional)
    scoreline_grid: list[ScorelineProb] | None = None

    # Contexto
    features: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def top_pick(self) -> MatchOutcome:
        if self.p_home >= self.p_draw and self.p_home >= self.p_away:
            return MatchOutcome.HOME
        if self.p_away >= self.p_draw and self.p_away >= self.p_home:
            return MatchOutcome.AWAY
        return MatchOutcome.DRAW
