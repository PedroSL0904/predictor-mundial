"""Grid search sobre los hiperparámetros del modelo.

Recorre un espacio de combinaciones de:
- elo_sigma: sensibilidad a la diferencia de Elo del rival
- recency_half_life_days: olvido exponencial
- shrinkage_matches: regularización bayesiana
- draw_penalty_threshold: gap mínimo para activar anti-1-1
- draw_penalty_strength: intensidad del anti-1-1
- elo_gap_inflation: factor de inflación de λ por gap Elo

Optimiza Brier promedio sobre los 3 Mundiales (2014, 2018, 2022).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv, normalize_team_name
from src.evaluation.backtest import (
    get_world_cup_matches,
    outcome_from_score,
)
from src.evaluation.backtest_elo import get_elo_at
from src.evaluation.metrics import summarize
from src.features.strengths import compute_weighted_strengths
from src.models import PoissonGoalModel, TeamStrength


def run_single_match(
    match: pd.Series,
    df: pd.DataFrame,
    timeline: dict,
    params: dict,
) -> tuple[tuple[float, float, float], str, tuple[int, int], tuple[int, int]] | None:
    """Predice un partido con params dados. Retorna (probs, outcome, pred_score, actual_score) o None si skip."""
    match_date = str(match["date"])[:10]
    train = df[df["date"] < match["date"]]
    if train.empty:
        return None

    home_norm = normalize_team_name(match["home_team"])
    away_norm = normalize_team_name(match["away_team"])

    elo_lookup = get_elo_at(timeline, match_date)
    strengths = compute_weighted_strengths(
        train,
        elo_lookup=elo_lookup,
        elo_sigma=params["elo_sigma"],
        recency_half_life_days=params["recency_half_life_days"],
        shrinkage_matches=params["shrinkage_matches"],
        min_weighted_matches=params["min_weighted_matches"],
    )
    h = strengths[strengths["team"] == home_norm]
    a = strengths[strengths["team"] == away_norm]
    if h.empty or a.empty:
        return None

    home = TeamStrength(
        name=home_norm,
        attack=float(h["attack"].iloc[0]),
        defense_vulnerability=float(h["defense_vulnerability"].iloc[0]),
    )
    away = TeamStrength(
        name=away_norm,
        attack=float(a["attack"].iloc[0]),
        defense_vulnerability=float(a["defense_vulnerability"].iloc[0]),
    )

    model = PoissonGoalModel(
        draw_penalty_threshold=params["draw_penalty_threshold"],
        draw_penalty_strength=params["draw_penalty_strength"],
        elo_gap_inflation=params["elo_gap_inflation"],
    )
    home_elo = elo_lookup.get(home_norm, 1500.0)
    away_elo = elo_lookup.get(away_norm, 1500.0)
    pred = model.predict(home, away, home_elo=home_elo, away_elo=away_elo)
    # silencio el backtest interno; ya tenemos el progreso a nivel grid

    return (
        (pred.p_home, pred.p_draw, pred.p_away),
        outcome_from_score(int(match["home_goals"]), int(match["away_goals"])),
        pred.most_likely_score,
        (int(match["home_goals"]), int(match["away_goals"])),
    )


def evaluate_params(
    df: pd.DataFrame,
    timeline: dict,
    params: dict,
    years: list[int],
) -> dict:
    """Evalúa un set de params sobre los Mundiales dados."""
    preds_all = []
    outs_all = []
    pred_scores = []
    actual_scores = []
    by_year = {}

    for year in years:
        wc = get_world_cup_matches(df, year)
        year_preds = []
        year_outs = []
        for _, match in wc.iterrows():
            res = run_single_match(match, df, timeline, params)
            if res is None:
                continue
            probs, outcome, p_score, a_score = res
            year_preds.append(probs)
            year_outs.append(outcome)
            preds_all.append(probs)
            outs_all.append(outcome)
            pred_scores.append(p_score)
            actual_scores.append(a_score)
        if year_preds:
            m = summarize(year_preds, year_outs)
            by_year[year] = m

    if not preds_all:
        return {"brier": 1.0, "params": params}

    overall = summarize(preds_all, outs_all, pred_scores, actual_scores)
    return {
        "brier": overall["brier"],
        "rps": overall["rps"],
        "log_loss": overall["log_loss"],
        "sign_accuracy": overall["sign_accuracy"],
        "exact_score_accuracy": overall["exact_score_accuracy"],
        "n": overall["n"],
        "by_year": by_year,
        "params": params,
    }


def grid_search(
    df: pd.DataFrame,
    timeline: dict,
    years: list[int] = [2014, 2018, 2022],
    fixed_params: dict | None = None,
) -> list[dict]:
    """Corre grid search sobre los hiperparámetros clave.

    Estrategia: random search sobre el espacio, ~50 combinaciones,
    quedándose con las top 10 por Brier.
    """
    fixed_params = fixed_params or {}
    base = {
        "elo_sigma": 200.0,
        "recency_half_life_days": 730.0,
        "shrinkage_matches": 10,
        "min_weighted_matches": 5.0,
        "draw_penalty_threshold": 0.05,
        "draw_penalty_strength": 0.15,
        "elo_gap_inflation": 0.08,
    }
    base.update(fixed_params)

    rng = np.random.default_rng(42)
    n_samples = 60

    # Espacio de búsqueda
    samples = []
    for _ in range(n_samples):
        p = {
            "elo_sigma": float(rng.choice([100, 150, 200, 250, 300, 400])),
            "recency_half_life_days": float(rng.choice([365, 540, 730, 1000, 1500, 2000])),
            "shrinkage_matches": int(rng.choice([3, 5, 8, 10, 15, 20, 30])),
            "min_weighted_matches": float(rng.choice([3, 5, 8, 10])),
            "draw_penalty_threshold": float(rng.choice([0.02, 0.04, 0.05, 0.08, 0.10])),
            "draw_penalty_strength": float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])),
            "elo_gap_inflation": float(rng.choice([0.0, 0.04, 0.08, 0.12, 0.16, 0.20])),
        }
        samples.append(p)

    # Incluir siempre el baseline (config actual)
    samples.insert(0, base)

    results = []
    total = len(samples)
    t0 = time.time()
    for i, params in enumerate(samples):
        if (i + 1) % 1 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            best_so_far = min((r["brier"] for r in results), default=1.0)
            print(
                f"  [{i+1}/{total}] elapsed={elapsed:.0f}s, eta={eta:.0f}s, best_brier={best_so_far:.4f}",
                flush=True,
            )
        m = evaluate_params(df, timeline, params, years)
        results.append(m)

    results.sort(key=lambda x: x["brier"])
    return results


def run_grid_search() -> list[dict]:
    """Helper entrypoint."""
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")
    print("Cargando timeline Elo...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    print(f"Timeline listo: {len(timeline)} fechas\n", flush=True)

    df = load_martj42_csv(csv_path)
    print("Iniciando grid search sobre 3 Mundiales...", flush=True)
    results = grid_search(df, timeline, years=[2014, 2018, 2022])

    print()
    print("=" * 90)
    print("TOP 10 CONFIGURACIONES POR BRIER")
    print("=" * 90)
    for r in results[:10]:
        p = r["params"]
        print(
            f"brier={r['brier']:.4f}  sign={r['sign_accuracy']:.1%}  "
            f"exact={r.get('exact_score_accuracy', 0):.1%}  "
            f"sigma={p['elo_sigma']:.0f}  rec={p['recency_half_life_days']:.0f}  "
            f"shr={p['shrinkage_matches']}  draw_thr={p['draw_penalty_threshold']:.2f}  "
            f"draw_str={p['draw_penalty_strength']:.2f}  elo_inf={p['elo_gap_inflation']:.2f}",
            flush=True,
        )

    # Guardar resultados
    out_path = Path("data/processed/grid_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        rr = {k: v for k, v in r.items() if k != "by_year"}
        rr["by_year_brier"] = {y: m["brier"] for y, m in r.get("by_year", {}).items()}
        rr["by_year_sign"] = {y: m["sign_accuracy"] for y, m in r.get("by_year", {}).items()}
        serializable.append(rr)
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nResultados guardados en {out_path}")

    return results


if __name__ == "__main__":
    run_grid_search()
