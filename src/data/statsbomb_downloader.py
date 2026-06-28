"""Downloader de StatsBomb open data.

Descarga matches + events de competitions especificas, extrae xG de los eventos,
y los almacena en data/raw/statsbomb_xg.json (mismo formato que el archivo
existente de WC 2018/2022).

StatsBomb open data:
    https://github.com/statsbomb/open-data

Estructura:
    data/matches/{competition_id}/{season_id}.json  -> lista de matches
    data/events/{match_id}.json                      -> eventos del match

xG por partido: suma de shot_statsbomb_xg de todos los shots del partido.

Competitions disponibles en open data:
    43/3   FIFA World Cup 2018
    43/106 FIFA World Cup 2022
    55/43  UEFA Euro 2020
    55/282 UEFA Euro 2024
    223/282 Copa America 2024
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from src.paths import RAW_DIR

STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
STATS_FILE = RAW_DIR / "statsbomb_xg.json"

DEFAULT_COMPETITIONS = [
    (43, 3),    # FIFA World Cup 2018
    (43, 106),  # FIFA World Cup 2022
    (55, 43),   # UEFA Euro 2020
    (55, 282),  # UEFA Euro 2024
    (223, 282),  # Copa America 2024
]


def _fetch(url: str, timeout: int = 30) -> requests.Response:
    """Fetch con retry simple."""
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def fetch_competition_matches(competition_id: int, season_id: int) -> list[dict]:
    """Descarga los matches metadata de una competition/season."""
    url = f"{STATSBOMB_BASE}/matches/{competition_id}/{season_id}.json"
    r = _fetch(url)
    return r.json()


def fetch_match_events(match_id: int) -> list[dict]:
    """Descarga los eventos de un match."""
    url = f"{STATSBOMB_BASE}/events/{match_id}.json"
    r = _fetch(url)
    return r.json()


def extract_match_xg(
    events: list[dict], home_team: str, away_team: str
) -> tuple[float, float]:
    """Calcula xG total por equipo para un match.

    Args:
        events: lista de eventos del match.
        home_team: nombre del equipo local (de match metadata).
        away_team: nombre del equipo visitante.

    Returns:
        (home_xg, away_xg).
    """
    home_xg = 0.0
    away_xg = 0.0
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        # StatsBomb almacena el xG en e["shot"]["statsbomb_xg"]
        # (sub-dict "shot"). Tambien hay un campo legacy
        # e["shot_statsbomb_xg"] que ya no se usa en data reciente.
        xg = (e.get("shot", {}) or {}).get("statsbomb_xg")
        if xg is None:
            xg = e.get("shot_statsbomb_xg", 0.0) or 0.0
        team = e.get("team", {}).get("name", "")
        if team == home_team:
            home_xg += xg
        elif team == away_team:
            away_xg += xg
    return home_xg, away_xg


def download_competition_xg(
    competition_id: int, season_id: int, verbose: bool = True
) -> dict[str, dict]:
    """Descarga todos los matches de una competition y extrae xG.

    Returns:
        dict {match_id_str: {home_team, away_team, date, home_xg, away_xg,
                              home_score, away_score}}.
    """
    if verbose:
        print(f"Fetching competition {competition_id}/{season_id}...", flush=True)
    matches = fetch_competition_matches(competition_id, season_id)
    if verbose:
        print(f"  {len(matches)} matches", flush=True)

    result: dict[str, dict] = {}
    for i, m in enumerate(matches):
        match_id = m["match_id"]
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        date = m["match_date"]
        try:
            events = fetch_match_events(match_id)
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(matches)}] skip match {match_id}: {e}", flush=True)
            continue
        h_xg, a_xg = extract_match_xg(events, home, away)
        # Scores: del metadata
        try:
            h_score = m["home_score"]
            a_score = m["away_score"]
        except KeyError:
            h_score = None
            a_score = None
        result[str(match_id)] = {
            "home_team": home,
            "away_team": away,
            "date": date,
            "home_xg": round(h_xg, 3),
            "away_xg": round(a_xg, 3),
            "home_score": h_score,
            "away_score": a_score,
        }
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(matches)}] processed", flush=True)
    return result


def merge_with_existing(
    new_data: dict[str, dict], existing_path: Path = STATS_FILE
) -> dict[str, dict]:
    """Merge new data con existing, sobreescribiendo si hay duplicados."""
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
    else:
        existing = {}
    existing.update(new_data)
    return existing


def main(
    competitions: list[tuple[int, int]] | None = None,
    output_path: Path = STATS_FILE,
) -> None:
    """Descarga todas las competitions y guarda en statsbomb_xg.json."""
    if competitions is None:
        competitions = DEFAULT_COMPETITIONS

    all_data: dict[str, dict] = {}
    for comp_id, season_id in competitions:
        new = download_competition_xg(comp_id, season_id)
        all_data.update(new)
        print(f"  -> {len(new)} matches con xG", flush=True)

    all_data = merge_with_existing(all_data, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nTotal matches con xG: {len(all_data)}")
    print(f"Guardado en {output_path}")


if __name__ == "__main__":
    main()
