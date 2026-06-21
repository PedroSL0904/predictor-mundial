"""Sistema de rating Elo rolling para selecciones nacionales.

Implementa un Elo estándar con:
- Origen 1500 para todos los equipos
- K=32 base, reducido a 24 en partidos de eliminatorias/torneo
  y elevado a 40 en goleadas (margen-of-victory multiplier)
- Ventaja de local embebida (no se usa cuando el partido es neutral)
- Bonus por diferencia de goles: G = ln(|diff| + 1)
- Expected score con ventaja local opcional
- Procesamiento partido a partido sobre CSV de martj42

El estado se persiste como JSON para reutilizar entre corridas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

# Constantes
ORIGINAL_ELO = 1500.0
BASE_K = 32.0
HOME_ADVANTAGE_ELO = 100.0  # bonus de local en escala Elo


class EloMatchUpdate(BaseModel):
    """Resultado de actualizar Elo después de un partido."""
    date: str
    home_team: str
    away_team: str
    home_elo_before: float
    away_elo_before: float
    home_elo_after: float
    away_elo_after: float
    expected_home: float
    expected_away: float
    k_factor: float


class EloRatingSystem:
    """Sistema de Elo rolling para partidos internacionales.

    Uso:
        elo = EloRatingSystem()
        elo.process_dataframe(df)  # df con columnas: date, home_team, away_team, home_goals, away_goals, neutral
        rating = elo.get_rating("Argentina")  # rating al final del histórico
    """

    def __init__(
        self,
        k_base: float = BASE_K,
        k_tournament: float = 24.0,
        home_advantage: float = HOME_ADVANTAGE_ELO,
        mov_multiplier: bool = True,
    ) -> None:
        self.ratings: dict[str, float] = {}
        self.k_base = k_base
        self.k_tournament = k_tournament
        self.home_advantage = home_advantage
        self.mov_multiplier = mov_multiplier
        self.history: list[EloMatchUpdate] = []

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, ORIGINAL_ELO)

    def set_rating(self, team: str, rating: float) -> None:
        self.ratings[team] = rating

    def expected_score(
        self, rating_a: float, rating_b: float, neutral: bool = True
    ) -> float:
        """Probabilidad esperada de que A gane (1 = seguro, 0 = imposible).

        Para partidos no-neutrales, el local recibe home_advantage puntos.
        """
        if not neutral:
            rating_a += self.home_advantage
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))

    def _margin_of_victory_mult(
        self, goal_diff: int, winner_elo: float, loser_elo: float
    ) -> float:
        """Multiplicador por margen de victoria (FIDE-style).

        Goleadas mueven más el rating, pero menos si el ganador ya era muy
        superior. Esto evita que un Germany 7-1 vs Curacao suba mucho el
        rating (ya era favorito).
        """
        if not self.mov_multiplier or goal_diff <= 1:
            return 1.0
        # Diferencia de rating normalizada
        elo_diff = abs(winner_elo - loser_elo)
        # Si el ganador era muy favorito, el bonus es menor
        favorite_damping = 1.0
        if elo_diff > 0:
            favorite_damping = 1.0 / (1.0 + elo_diff / 800.0)
        return math.log(goal_diff + 1) * favorite_damping

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        neutral: bool = True,
        tournament: str | None = None,
        date: str | None = None,
    ) -> EloMatchUpdate:
        """Procesa un partido y actualiza los Elo."""
        r_h = self.get_rating(home_team)
        r_a = self.get_rating(away_team)

        e_h = self.expected_score(r_h, r_a, neutral=neutral)
        e_a = 1.0 - e_h

        # Resultado real (1 = gana, 0.5 = empate, 0 = pierde)
        if home_goals > away_goals:
            s_h, s_a = 1.0, 0.0
            goal_diff = home_goals - away_goals
            winner_elo, loser_elo = r_h, r_a
        elif home_goals < away_goals:
            s_h, s_a = 0.0, 1.0
            goal_diff = away_goals - home_goals
            winner_elo, loser_elo = r_a, r_h
        else:
            s_h, s_a = 0.5, 0.5
            goal_diff = 0
            winner_elo = loser_elo = 0  # no aplica

        # K-factor: menor en partidos de torneo (más estables)
        k = self.k_base
        if tournament and ("World Cup" in tournament or "Euro" in tournament or "Copa" in tournament):
            k = self.k_tournament
        # Bonus por margen
        if self.mov_multiplier and goal_diff > 0:
            k *= self._margin_of_victory_mult(goal_diff, winner_elo, loser_elo)

        new_r_h = r_h + k * (s_h - e_h)
        new_r_a = r_a + k * (s_a - e_a)

        self.set_rating(home_team, new_r_h)
        self.set_rating(away_team, new_r_a)

        update = EloMatchUpdate(
            date=date or "",
            home_team=home_team,
            away_team=away_team,
            home_elo_before=r_h,
            away_elo_before=r_a,
            home_elo_after=new_r_h,
            away_elo_after=new_r_a,
            expected_home=e_h,
            expected_away=e_a,
            k_factor=k,
        )
        self.history.append(update)
        return update

    def process_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        home_col: str = "home_team",
        away_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        neutral_col: str = "neutral_venue",
        tournament_col: str = "tournament",
    ) -> None:
        """Procesa todos los partidos del DataFrame en orden cronológico."""
        # Filtrar partidos sin goles (futuros o sin datos)
        df_clean = df.dropna(subset=[home_goals_col, away_goals_col]).copy()
        df_clean[home_goals_col] = df_clean[home_goals_col].astype(int)
        df_clean[away_goals_col] = df_clean[away_goals_col].astype(int)

        df_sorted = df_clean.sort_values(date_col).reset_index(drop=True)
        for _, row in df_sorted.iterrows():
            self.update(
                home_team=row[home_col],
                away_team=row[away_col],
                home_goals=int(row[home_goals_col]),
                away_goals=int(row[away_goals_col]),
                neutral=bool(row[neutral_col]),
                tournament=row.get(tournament_col),
                date=str(row[date_col]),
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ratings": self.ratings,
            "config": {
                "k_base": self.k_base,
                "k_tournament": self.k_tournament,
                "home_advantage": self.home_advantage,
                "mov_multiplier": self.mov_multiplier,
            },
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.ratings = data["ratings"]
        cfg = data.get("config", {})
        self.k_base = cfg.get("k_base", self.k_base)
        self.k_tournament = cfg.get("k_tournament", self.k_tournament)
        self.home_advantage = cfg.get("home_advantage", self.home_advantage)
        self.mov_multiplier = cfg.get("mov_multiplier", self.mov_multiplier)


def build_elo_table(df: pd.DataFrame, cache_path: Path | None = None) -> EloRatingSystem:
    """Helper: procesa el CSV de martj42 y devuelve EloRatingSystem.

    Si cache_path existe, lo carga en vez de reprocesar.
    """
    if cache_path and cache_path.exists():
        elo = EloRatingSystem()
        elo.load(cache_path)
        return elo

    elo = EloRatingSystem()
    elo.process_dataframe(df)
    if cache_path:
        elo.save(cache_path)
    return elo


def current_top_n(elo: EloRatingSystem, n: int = 20) -> list[tuple[str, float]]:
    """Devuelve los N equipos mejor rankeados."""
    sorted_teams = sorted(elo.ratings.items(), key=lambda x: -x[1])
    return sorted_teams[:n]
