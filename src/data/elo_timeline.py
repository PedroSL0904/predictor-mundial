"""Construye un cache incremental de Elo pre-partido para acelerar el backtest.

Para cada fecha única en el dataset, guardamos el dict de Elo resultante.
El backtest solo necesita el Elo justo antes de cada partido del Mundial,
así que en vez de reprocesar 49k partidos por cada partido del WC, podemos
precomputar el Elo una vez para todas las fechas.

Esto reduce el backtest de horas a minutos.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.elo import EloRatingSystem
from src.data.historical import load_martj42_csv


def build_elo_timeline(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Procesa todos los partidos UNA vez, devolviendo el Elo post-partido.

    Retorna dict fecha_iso -> {team: elo}.
    Solo guardamos las fechas que son "hitos" (1 por día o por partido).

    Para el backtest, el Elo "pre-partido" de un partido en fecha D
    es el Elo del último partido en fecha < D.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    # Filtrar NaN en goles
    df_sorted = df_sorted.dropna(subset=["home_goals", "away_goals"])
    df_sorted["home_goals"] = df_sorted["home_goals"].astype(int)
    df_sorted["away_goals"] = df_sorted["away_goals"].astype(int)

    elo = EloRatingSystem()
    timeline: dict[str, dict[str, float]] = {}

    total = len(df_sorted)
    last_pct = -1
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        if i % 2000 == 0:
            pct = int(100 * i / total)
            if pct != last_pct:
                print(f"  Elo timeline: {pct}% ({i}/{total} partidos)", flush=True)
                last_pct = pct
        date_iso = str(row["date"])[:10]
        # Guardar snapshot ANTES de este partido
        timeline[date_iso] = dict(elo.ratings)
        # Procesar partido
        elo.update(
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_goals=int(row["home_goals"]),
            away_goals=int(row["away_goals"]),
            neutral=bool(row.get("neutral_venue", True)),
            tournament=row.get("tournament"),
            date=date_iso,
        )

    return timeline


def get_elo_before(timeline: dict[str, dict[str, float]], as_of_date: str) -> dict[str, float]:
    """Devuelve el Elo snapshot más reciente antes (o igual) a as_of_date.

    Estrategia: scan lineal de fechas conocidas ≤ as_of_date.
    Como el dataset está ordenado, podemos ser eficientes.
    """
    # Timeline guarda snapshots por fecha. Buscar la fecha más reciente ≤ target.
    candidates = [d for d in timeline if d <= as_of_date]
    if not candidates:
        return {}
    latest = max(candidates)
    return timeline[latest]


def precompute_and_cache(csv_path: Path, cache_path: Path) -> dict[str, dict[str, float]]:
    """Construye el timeline y lo guarda como JSON."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    df = load_martj42_csv(csv_path)
    timeline = build_elo_timeline(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(timeline))
    return timeline


if __name__ == "__main__":
    import time
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")
    t0 = time.time()
    timeline = precompute_and_cache(csv_path, cache_path)
    print(f"Timeline con {len(timeline)} fechas únicas en {time.time() - t0:.1f}s")
    print(f"Cache guardado en {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")
    # Spot check
    keys = sorted(timeline.keys())
    print(f"Rango fechas: {keys[0]} a {keys[-1]}")
    print(f"Equipos en última fecha: {len(timeline[keys[-1]])}")
