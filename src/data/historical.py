"""Carga de datos históricos de partidos internacionales.

Fuentes soportadas:
- CSV de football-data.co.uk (gratis, resultados + cuotas)
- CSV del repo martj42/international_results (49k partidos desde 1872)
- xG de Understat (cuando la API esté disponible)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain import MatchResult

MARTJ42_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/"
    "results.csv"
)
FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/"


def load_martj42_csv(path: Path | str) -> pd.DataFrame:
    """Carga el CSV de martj42/international_results.

    Columnas: date, home_team, away_team, home_score, away_score,
              tournament, city, country, neutral.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={
        "home_score": "home_goals",
        "away_score": "away_goals",
        "neutral": "neutral_venue",
    })
    df["neutral_venue"] = df["neutral_venue"].astype(bool)
    return df


def download_martj42_csv(dest: Path) -> Path:
    """Descarga el CSV de martj42 a dest. Retorna el path."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        r = requests.get(MARTJ42_URL, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def filter_by_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Filtra partidos de los últimos N años."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * years)
    return df[df["date"] >= cutoff].copy()


def filter_by_tournaments(
    df: pd.DataFrame, tournaments: list[str] | None = None
) -> pd.DataFrame:
    """Filtra por nombre de torneo (case-insensitive contains)."""
    if not tournaments:
        return df
    pattern = "|".join(tournaments)
    return df[df["tournament"].str.contains(pattern, case=False, na=False)].copy()


def to_match_results(df: pd.DataFrame) -> list[MatchResult]:
    """Convierte DataFrame a lista de MatchResult."""
    return [
        MatchResult(
            home_team=row["home_team"],
            away_team=row["away_team"],
            date=row["date"].to_pydatetime() if hasattr(row["date"], "to_pydatetime") else row["date"],
            home_goals=int(row["home_goals"]),
            away_goals=int(row["away_goals"]),
            neutral_venue=bool(row.get("neutral_venue", True)),
            tournament=row.get("tournament"),
        )
        for _, row in df.iterrows()
    ]


def compute_strengths_from_results(
    df: pd.DataFrame,
    min_matches: int = 5,
) -> pd.DataFrame:
    """Calcula attack/defense strength desde goles reales.

    attack(team) = promedio goles a favor por partido
    defense_vulnerability(team) = promedio goles en contra por partido
    """
    if df.empty:
        return pd.DataFrame()

    home = df.groupby("home_team").agg(
        gf=("home_goals", "mean"),
        ga=("away_goals", "mean"),
        matches=("home_goals", "count"),
    ).rename_axis("team").reset_index()

    away = df.groupby("away_team").agg(
        gf=("away_goals", "mean"),
        ga=("home_goals", "mean"),
        matches=("away_goals", "count"),
    ).rename_axis("team").reset_index()

    combined = pd.concat([home, away]).groupby("team").agg(
        attack=("gf", "mean"),
        defense_vulnerability=("ga", "mean"),
        matches=("matches", "sum"),
    ).reset_index()

    combined = combined[combined["matches"] >= min_matches]
    return combined.sort_values("attack", ascending=False).reset_index(drop=True)


# Mapeo de nombres alternativos (Understat/FBref → football-data.co.uk / martj42)
# Delegado a src.data.team_names (un solo lugar para mantener consistencia).
from src.data.team_names import normalize_team_name  # noqa: F401
