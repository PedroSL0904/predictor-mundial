"""Scraper de cuotas de mercado (Pinnacle via The Odds API).

The Odds API: https://the-odds-api.com/
- Free tier: 500 requests/mes
- Devuelve cuotas decimal de múltiples bookmakers
- Filtra por 'h2h' (1X2) y region 'eu' para tener Pinnacle

Alternativa: football-data.co.uk provee CSVs históricos con cuotas
de múltiples bookmakers, útil para backtesting.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "eu"
DEFAULT_MARKETS = "h2h"
DEFAULT_BOOKMAKERS = "pinnacle"


class OddsQuote(BaseModel):
    """Cuotas de un partido."""

    home_team: str
    away_team: str
    commence_time: datetime
    bookmaker: str
    p_home: float
    p_draw: float
    p_away: float
    raw: dict | None = None


class OddsAPIClient:
    """Cliente para The Odds API."""

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("data/raw/odds")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_soccer_odds(
        self,
        sport: str = "soccer",
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
        bookmakers: str | None = DEFAULT_BOOKMAKERS,
        use_cache: bool = True,
    ) -> list[dict]:
        """Baja cuotas actuales de fútbol.

        sport puede ser 'soccer' (todos), 'soccer_fifa_world_cup' o
        'soccer_international'.
        """
        cache_file = self.cache_dir / f"{sport}_{datetime.now().strftime('%Y%m%d')}.json"
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())

        if not self.api_key:
            print("WARN: No hay ODDS_API_KEY. Devolviendo lista vacía.")
            return []

        url = f"{ODDS_API_BASE}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            cache_file.write_text(json.dumps(data, default=str))
            return data
        except requests.RequestException as e:
            print(f"Error fetching odds: {e}")
            return []

    def parse_quotes(self, raw_data: list[dict]) -> list[OddsQuote]:
        """Parsea respuesta de The Odds API a OddsQuote."""
        quotes = []
        for event in raw_data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            commence = event.get("commence_time", "")
            for book in event.get("bookmakers", []):
                book_key = book.get("key", "")
                for market in book.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    p_h = 1.0 / outcomes.get(home, 1e9)
                    p_d = 1.0 / outcomes.get("Draw", 1e9)
                    p_a = 1.0 / outcomes.get(away, 1e9)
                    total = p_h + p_d + p_a
                    # Normalizar para quitar el overround (vig)
                    if total > 0:
                        p_h /= total
                        p_d /= total
                        p_a /= total
                    quotes.append(OddsQuote(
                        home_team=home,
                        away_team=away,
                        commence_time=commence,
                        bookmaker=book_key,
                        p_home=p_h,
                        p_draw=p_d,
                        p_away=p_a,
                        raw=event,
                    ))
        return quotes

    def fetch_and_parse(
        self,
        sport: str = "soccer_fifa_world_cup",
    ) -> list[OddsQuote]:
        raw = self.fetch_soccer_odds(sport=sport)
        return self.parse_quotes(raw)


def football_data_odds_csv_url(season: str) -> str:
    """Devuelve URL de CSVs con cuotas históricas de football-data.co.uk.

    Para Inglaterra: https://www.football-data.co.uk/mmz4281/2425/E0.csv
    """
    # Por ahora solo soportamos ligas top; para internacionales hay que
    # usar otra fuente. La dejo como placeholder.
    raise NotImplementedError(
        "Football-data.co.uk no tiene archivo dedicado a partidos de "
        "selección. Usar oddsportal.com scrapeado o API-Football."
    )


def implied_probs_from_decimal(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """Convierte cuotas decimales a probabilidades implícitas normalizadas (sin vig)."""
    p_h = 1.0 / odds_h
    p_d = 1.0 / odds_d
    p_a = 1.0 / odds_a
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total
