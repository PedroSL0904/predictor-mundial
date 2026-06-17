"""Descarga xG de StatsBomb open data para WC 2018 y 2022.

Output: data/raw/statsbomb_xg.json con formato:
{
  "match_id": {
    "home_team": "France",
    "away_team": "Croatia",
    "date": "2018-07-15",
    "home_xg": 2.34,
    "away_xg": 1.87,
    "home_score": 4,
    "away_score": 2
  }
}
"""
import json
import time
from pathlib import Path

import requests


CACHE_DIR = Path("data/raw/statsbomb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = Path("data/raw/statsbomb_xg.json")

SEASONS = {
    3: "FIFA World Cup 2018",
    106: "FIFA World Cup 2022",
}


def fetch(url, retries=3, timeout=30):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2)


def parse_match(match_meta, events):
    """Suma xG de shots por equipo en este partido."""
    home = match_meta["home_team"]["home_team_name"]
    away = match_meta["away_team"]["away_team_name"]
    date = match_meta["match_date"]
    home_score = match_meta.get("home_score", 0)
    away_score = match_meta.get("away_score", 0)

    home_xg = 0.0
    away_xg = 0.0
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        shot = e.get("shot", {})
        xg = shot.get("statsbomb_xg")
        if xg is None:
            continue
        team = e.get("team", {}).get("name")
        if team == home:
            home_xg += xg
        elif team == away:
            away_xg += xg
    return {
        "home_team": home,
        "away_team": away,
        "date": date,
        "home_xg": round(home_xg, 3),
        "away_xg": round(away_xg, 3),
        "home_score": home_score,
        "away_score": away_score,
    }


def main():
    matches_meta = {}
    for season_id, name in SEASONS.items():
        url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/{season_id}.json"
        print(f"Descargando metadata {name}...", flush=True)
        r = fetch(url)
        ms = r.json()
        matches_meta.update({m["match_id"]: m for m in ms})
        print(f"  {len(ms)} partidos", flush=True)

    print(f"\nTotal partidos a procesar: {len(matches_meta)}", flush=True)
    print(f"Descargando eventos (puede tardar 5-10 min)...\n", flush=True)

    results = {}
    t0 = time.time()
    completed = 0
    for mid, meta in matches_meta.items():
        cache_file = CACHE_DIR / f"{mid}.json"
        try:
            if cache_file.exists():
                with open(cache_file) as f:
                    events = json.load(f)
            else:
                url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{mid}.json"
                r = fetch(url, timeout=60)
                events = r.json()
                with open(cache_file, "w") as f:
                    json.dump(events, f)
            parsed = parse_match(meta, events)
            results[mid] = parsed
            completed += 1
            if completed % 5 == 0 or completed == len(matches_meta):
                elapsed = time.time() - t0
                eta = elapsed / completed * (len(matches_meta) - completed)
                print(f"  [{completed}/{len(matches_meta)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        except Exception as e:
            print(f"  ERROR en match {mid}: {e}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{len(results)} partidos guardados en {OUT_PATH}")
    print(f"Tiempo total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
