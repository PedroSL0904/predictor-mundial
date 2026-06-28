"""Loader de xG real desde StatsBomb open data.

Cache local en data/raw/statsbomb_xg.json (descargado por download_statsbomb.py).

Solo disponible para WC 2018 y WC 2022 (lo que StatsBomb libera).
Para el resto de partidos, hay que usar la aproximacion por Elo
(_approx_xg_from_elo en strengths.py).

API:
    from src.data.statsbomb import get_xg_real_lookup
    lookup = get_xg_real_lookup()
    # lookup[(date_iso, home_team, away_team)] = (home_xg, away_xg)
    # Retorna None si no hay xG real para ese partido
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

STATSBOMB_PATH = Path("data/raw/statsbomb_xg.json")


@lru_cache(maxsize=1)
def _load_xg_raw() -> dict:
    if not STATSBOMB_PATH.exists():
        return {}
    with open(STATSBOMB_PATH) as f:
        return json.load(f)


def _normalize_team(name: str) -> str:
    """Normaliza nombres StatsBomb -> nombres de martj42/international_results.

    StatsBomb usa los nombres FIFA oficiales. martj42 también, pero hay
    algunas diferencias (espacios, tildes, nombres largos).
    """
    replacements = {
        "Curaçao": "Curaçao",
        "Curacao": "Curaçao",
        "Korea Republic": "South Korea",
        "IR Iran": "Iran",
        "USA": "United States",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Cöte d'Ivoire": "Ivory Coast",
        "Czechia": "Czech Republic",
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
        "Cape Verde Islands": "Cape Verde",
        "Cape Verde": "Cape Verde",
        "DR Congo": "DR Congo",
        "Congo DR": "DR Congo",
        "Democratic Republic of Congo": "DR Congo",
    }
    return replacements.get(name, name)


def get_xg_real_lookup() -> dict[tuple[str, str, str], tuple[float, float]]:
    """Devuelve dict {(date_iso, home_norm, away_norm): (home_xg, away_xg)}.

    Las claves usan nombres normalizados a martj42 para que el backtest
    pueda cruzar con el dataset principal.

    Cacheado: el dict procesado se cachea con lru_cache para evitar
    reconstruir ~125 entries en cada llamada (costo: ~50-100ms).
    """
    raw = _load_xg_raw()
    return {
        (m["date"], _normalize_team(m["home_team"]), _normalize_team(m["away_team"])):
        (m["home_xg"], m["away_xg"])
        for m in raw.values()
    }


# Alias interno con cache para evitar reconstruir el dict
_cached_xg_lookup = lru_cache(maxsize=1)(get_xg_real_lookup)


def get_xg_real_lookup_cached() -> dict[tuple[str, str, str], tuple[float, float]]:
    """Versión cacheada de get_xg_real_lookup(). Usar en código hot-path."""
    return _cached_xg_lookup()


def has_xg_real(date_iso: str, home_team: str, away_team: str) -> bool:
    """Chequea si hay xG real disponible para este partido."""
    h = _normalize_team(home_team)
    a = _normalize_team(away_team)
    return (date_iso, h, a) in get_xg_real_lookup()


def get_xg_real(date_iso: str, home_team: str, away_team: str) -> tuple[float, float] | None:
    """Devuelve (home_xg, away_xg) o None si no hay xG real."""
    h = _normalize_team(home_team)
    a = _normalize_team(away_team)
    return get_xg_real_lookup().get((date_iso, h, a))


def statsbomb_coverage(df) -> dict[str, int]:
    """Para un DataFrame con partidos, cuenta cuantos tienen xG real disponible.

    Devuelve dict {"total": n, "with_xg": m, "by_year": {year: count}}.
    """
    lookup = get_xg_real_lookup()
    by_year: dict[str, int] = {}
    total_with = 0
    for _, m in df.iterrows():
        date_iso = str(m["date"])[:10]
        h = _normalize_team(m["home_team"])
        a = _normalize_team(m["away_team"])
        if (date_iso, h, a) in lookup:
            total_with += 1
            y = date_iso[:4]
            by_year[y] = by_year.get(y, 0) + 1
    return {"total": len(df), "with_xg": total_with, "by_year": by_year}
