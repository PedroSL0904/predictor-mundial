"""Scraper de Understat para datos de xG de selecciones nacionales.

Usa la lib `understatapi` que wrappea los endpoints internos de understat.com.
Para partidos de selección, Understat solo tiene los de competencias
internacionales (torneos y amistosos desde 2014+).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

try:
    from understatapi import UnderstatClient
    UNDERSTAT_AVAILABLE = True
except ImportError:
    UNDERSTAT_AVAILABLE = False


class UnderstatMatch(BaseModel):
    """Partido de Understat normalizado."""

    match_id: str
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    tournament: str | None = None


class UnderstatFetcher:
    """Cliente de Understat para selecciones nacionales.

    Understat no expone una "selección" como tal: hay que buscar por
    el equipo (Argentina, Brazil, etc.) y filtrar partidos de selección.
    """

    # Equipos del Mundial 2026 (48 selecciones)
    WC_2026_TEAMS = [
        "Argentina", "Brazil", "France", "England", "Spain", "Portugal",
        "Netherlands", "Belgium", "Germany", "Italy", "Croatia", "Uruguay",
        "Mexico", "USA", "Canada", "Japan", "South Korea", "Australia",
        "Morocco", "Senegal", "Nigeria", "Ghana", "Tunisia", "Algeria",
        "Egypt", "Cameroon", "Ivory Coast", "Iran", "Saudi Arabia", "Qatar",
        "Iraq", "Jordan", "Uzbekistan", "Ecuador", "Colombia", "Chile",
        "Peru", "Paraguay", "Panama", "Costa Rica", "Haiti", "Curacao",
        "Cape Verde", "South Africa", "Norway", "Sweden", "Switzerland",
        "Austria", "Scotland", "Bosnia and Herzegovina", "New Zealand",
        "Czechia", "Congo DR",
    ]

    def __init__(self, cache_dir: Path | None = None) -> None:
        if not UNDERSTAT_AVAILABLE:
            raise ImportError(
                "understatapi no instalado. Ejecutá: pip install understatapi"
            )
        self.client = UnderstatClient()
        self.cache_dir = cache_dir or Path("data/raw/understat")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, team: str) -> Path:
        safe = team.lower().replace(" ", "_")
        return self.cache_dir / f"{safe}_matches.json"

    def fetch_team_matches(self, team: str, use_cache: bool = True) -> list[dict]:
        """Baja todos los partidos de un equipo."""
        cache = self._cache_path(team)
        if use_cache and cache.exists():
            return json.loads(cache.read_text())

        try:
            # understatapi usa season (año) y team
            all_matches = []
            for year in range(2014, 2026):
                try:
                    matches = self.client.team(team).get_team_results(year)
                    all_matches.extend(matches)
                except Exception:
                    continue
                time.sleep(0.3)  # rate limit suave
        except Exception as e:
            print(f"Error fetch {team}: {e}")
            return []

        cache.write_text(json.dumps(all_matches))
        return all_matches

    def fetch_all_wc_teams(
        self, use_cache: bool = True, delay: float = 0.5
    ) -> dict[str, list[dict]]:
        """Baja partidos de todas las selecciones del Mundial 2026."""
        out = {}
        for i, team in enumerate(self.WC_2026_TEAMS):
            print(f"[{i + 1}/{len(self.WC_2026_TEAMS)}] Fetching {team}...")
            out[team] = self.fetch_team_matches(team, use_cache=use_cache)
            time.sleep(delay)
        return out

    def to_dataframe(self, raw_matches: list[dict]) -> pd.DataFrame:
        """Convierte partidos crudos de Understat a DataFrame limpio."""
        if not raw_matches:
            return pd.DataFrame()

        rows = []
        for m in raw_matches:
            try:
                sides = m.get("side", "")
                h = m.get("h", {})
                a = m.get("a", {})
                if not h or not a:
                    continue
                rows.append({
                    "date": m.get("datetime", "")[:10],
                    "home_team": h.get("title", ""),
                    "away_team": a.get("title", ""),
                    "home_goals": int(h.get("goals", 0)),
                    "away_goals": int(a.get("goals", 0)),
                    "home_xg": float(h.get("xG", 0.0)),
                    "away_xg": float(a.get("xG", 0.0)),
                    "tournament": m.get("competition", ""),
                })
            except Exception:
                continue

        return pd.DataFrame(rows)


def compute_team_strengths(matches: pd.DataFrame) -> pd.DataFrame:
    """Calcula attack/defense strength de cada equipo a partir de xG.

    attack = xG promedio a favor por partido
    defense_vulnerability = xG promedio en contra por partido

    Devuelve DataFrame con columnas: team, attack, defense_vulnerability, matches.
    """
    if matches.empty:
        return pd.DataFrame()

    # Local
    home = matches.groupby("home_team").agg(
        gf_xg=("home_xg", "mean"),
        ga_xg=("away_xg", "mean"),
        matches=("home_xg", "count"),
    ).rename_axis("team").reset_index()

    # Visitante
    away = matches.groupby("away_team").agg(
        gf_xg=("away_xg", "mean"),
        ga_xg=("home_xg", "mean"),
        matches=("away_xg", "count"),
    ).rename_axis("team").reset_index()

    # Unir y promediar ponderado por cantidad de partidos
    combined = pd.concat([home, away]).groupby("team").agg(
        attack=("gf_xg", "mean"),
        defense_vulnerability=("ga_xg", "mean"),
        matches=("matches", "sum"),
    ).reset_index()

    return combined.sort_values("attack", ascending=False)
