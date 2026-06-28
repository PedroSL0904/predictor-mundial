"""Construye un cache incremental de Elo pre-partido para acelerar el backtest.

Para cada fecha única en el dataset, guardamos el dict de Elo resultante.
El backtest solo necesita el Elo justo antes de cada partido del Mundial,
así que en vez de reprocesar 49k partidos por cada partido del WC, podemos
precomputar el Elo una vez para todas las fechas.

Esto reduce el backtest de horas a minutos.

El cache se persiste como **Parquet** columnar (en vez de JSON nested dict).
Parquet es ~30x más pequeño y ~3-5x más rápido de cargar que JSON.
Formato interno: dict estándar fecha_iso -> {team: elo} (mantiene compat
con todo el código existente).
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import pandas as pd

from src.data.elo import EloRatingSystem
from src.data.historical import load_martj42_csv
from src.logging_config import get_logger

logger = get_logger(__name__)

_PARQUET_CACHE_NAME = "elo_timeline.parquet"
_JSON_CACHE_NAME = "elo_timeline.json"


def build_elo_timeline(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Procesa todos los partidos UNA vez, devolviendo el Elo post-partido.

    Retorna dict fecha_iso -> {team: elo}.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
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
                logger.info(f"  Elo timeline: {pct}% ({i}/{total} partidos)")
                last_pct = pct
        date_iso = str(row["date"])[:10]
        timeline[date_iso] = dict(elo.ratings)
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


def get_elo_at(timeline: dict[str, dict[str, float]], as_of_date: str) -> dict[str, float]:
    """Snapshot de Elo más reciente en o antes de `as_of_date`.

    O(log n) con bisect usando sorted_dates cacheado en el dict.
    El dict acepta la key mágica "__sorted_dates__" que es la lista ordenada.
    """
    if not timeline:
        return {}
    cached = timeline.get("__sorted_dates__")
    if cached is None:
        cached = sorted(k for k in timeline.keys() if not k.startswith("__"))
        timeline["__sorted_dates__"] = cached
    idx = bisect.bisect_right(cached, as_of_date) - 1
    if idx < 0:
        return {}
    return timeline[cached[idx]]


# Alias para compatibilidad
get_elo_before = get_elo_at


def _timeline_to_parquet(timeline: dict[str, dict[str, float]], path: Path) -> None:
    """Convierte dict[date, {team: elo}] a DataFrame long y guarda como Parquet."""
    rows = []
    for date_iso, ratings in timeline.items():
        if date_iso.startswith("__"):
            continue
        for team, elo in ratings.items():
            rows.append({"date": date_iso, "team": team, "elo": float(elo)})
    df = pd.DataFrame(rows, columns=["date", "team", "elo"])
    df["date"] = df["date"].astype(str)
    df.to_parquet(path, index=False, compression="snappy")


def _parquet_to_timeline(path: Path) -> dict[str, dict[str, float]]:
    """Carga Parquet y reconstruye dict[date, {team: elo}] usando groupby."""
    df = pd.read_parquet(path, columns=["date", "team", "elo"])
    # groupby + agg(list-of-tuples) es más rápido que iterrows
    grouped = df.groupby("date", sort=True)
    timeline: dict[str, dict[str, float]] = {}
    for date_iso, group in grouped:
        timeline[str(date_iso)[:10]] = dict(
            zip(group["team"].tolist(), group["elo"].tolist(), strict=True)
        )
    return timeline


def precompute_and_cache(
    csv_path: Path,
    cache_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Construye el timeline y lo guarda como Parquet.

    Args:
        csv_path: ruta al CSV de martj42.
        cache_path: ruta al cache (.parquet). Si None, usa
            data/processed/elo_timeline.parquet.

    Returns:
        dict fecha_iso -> {team: elo}

    Si existe un JSON legacy, lo migra automáticamente a Parquet.
    """
    if cache_path is None:
        cache_path = Path("data/processed") / _PARQUET_CACHE_NAME

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = (
        cache_path.with_suffix(".parquet") if cache_path.suffix == ".json" else cache_path
    )

    if parquet_path.exists():
        return _parquet_to_timeline(parquet_path)

    json_path = parquet_path.parent / _JSON_CACHE_NAME
    if json_path.exists() and parquet_path.suffix == ".parquet":
        logger.info(f"Migrando {json_path.name} -> {parquet_path.name}...")
        timeline = json.loads(json_path.read_text())
        _timeline_to_parquet(timeline, parquet_path)
        # Recargar desde Parquet para garantizar consistencia
        return _parquet_to_timeline(parquet_path)

    df = load_martj42_csv(csv_path)
    timeline = build_elo_timeline(df)

    if parquet_path.suffix == ".parquet":
        _timeline_to_parquet(timeline, parquet_path)
        return _parquet_to_timeline(parquet_path)
    else:
        parquet_path.write_text(json.dumps(timeline))
        return timeline


if __name__ == "__main__":
    import time
    csv_path = Path("data/raw/martj42_results.csv")
    t0 = time.time()
    timeline = precompute_and_cache(csv_path)
    elapsed = time.time() - t0
    logger.info(f"Timeline con {len(timeline)} fechas únicas en {elapsed:.1f}s")
    parquet = Path("data/processed/elo_timeline.parquet")
    if parquet.exists():
        logger.info(f"Cache guardado en {parquet} ({parquet.stat().st_size / 1e6:.1f} MB)")
    keys = sorted(k for k in timeline.keys() if not k.startswith("__"))
    logger.info(f"Rango fechas: {keys[0]} a {keys[-1]}")
    logger.info(f"Equipos en última fecha: {len(timeline[keys[-1]])}")
