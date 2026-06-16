"""Backtest que compara modelo baseline vs modelo con Elo ponderado.

El backtest usa el timeline de Elo precomputado (ver elo_timeline.py)
para que sea rápido: una sola pasada construye el Elo para todas las
fechas y después cada partido del Mundial solo consulta el snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import precompute_and_cache
from src.data.historical import (
    compute_strengths_from_results,
    load_martj42_csv,
    normalize_team_name,
)
from src.evaluation.backtest import (
    get_world_cup_matches,
    outcome_from_score,
)
from src.evaluation.metrics import summarize
from src.features.strengths import compute_weighted_strengths
from src.models import PoissonGoalModel, TeamStrength


def get_elo_at(timeline: dict[str, dict[str, float]], as_of: str) -> dict[str, float]:
    """Snapshot de Elo más reciente en o antes de as_of."""
    candidates = [d for d in timeline if d <= as_of]
    if not candidates:
        return {}
    return timeline[max(candidates)]


def backtest_strategy(
    df: pd.DataFrame,
    year: int,
    timeline: dict[str, dict[str, float]],
    strategy: str = "elo_weighted",
    min_weighted_matches: float = 5.0,
    verbose: bool = True,
) -> dict:
    """Corre el backtest de un Mundial con la estrategia dada.

    Strategies:
    - "baseline": attack/defense con promedio crudo de goles
    - "elo_weighted": attack/defense con ponderación por Elo del rival
    """
    wc = get_world_cup_matches(df, year)
    if wc.empty:
        return {"year": year, "strategy": strategy, "n": 0}

    model = PoissonGoalModel()
    predictions = []
    outcomes = []
    predicted_scores = []
    actual_scores = []
    skipped = 0

    total = len(wc)
    for i, (_, match) in enumerate(wc.iterrows()):
        if verbose and i % 10 == 0:
            print(f"    [{i}/{total}]", end=" ", flush=True)
        match_date = str(match["date"])[:10]
        train = df[df["date"] < match["date"]].copy()
        if train.empty:
            skipped += 1
            continue

        home_norm = normalize_team_name(match["home_team"])
        away_norm = normalize_team_name(match["away_team"])

        if strategy == "baseline":
            strengths = compute_strengths_from_results(train, min_matches=5)
            h = strengths[strengths["team"] == home_norm]
            a = strengths[strengths["team"] == away_norm]
        elif strategy == "elo_weighted":
            elo_lookup = get_elo_at(timeline, match_date)
            strengths = compute_weighted_strengths(
                train,
                elo_lookup=elo_lookup,
                elo_sigma=200.0,
                recency_half_life_days=730.0,
                shrinkage_matches=10,
                min_weighted_matches=min_weighted_matches,
            )
            h = strengths[strengths["team"] == home_norm]
            a = strengths[strengths["team"] == away_norm]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if h.empty or a.empty:
            skipped += 1
            continue

        home = TeamStrength(
            name=home_norm,
            attack=float(h["attack"].iloc[0]),
            defense_vulnerability=float(h["defense_vulnerability"].iloc[0]),
            matches=int(h["matches"].iloc[0]),
        )
        away = TeamStrength(
            name=away_norm,
            attack=float(a["attack"].iloc[0]),
            defense_vulnerability=float(a["defense_vulnerability"].iloc[0]),
            matches=int(a["matches"].iloc[0]),
        )

        elo_lookup = get_elo_at(timeline, match_date)
        home_elo = elo_lookup.get(home_norm, ORIGINAL_ELO)
        away_elo = elo_lookup.get(away_norm, ORIGINAL_ELO)

        pred = model.predict(home, away, home_elo=home_elo, away_elo=away_elo)
        predictions.append((pred.p_home, pred.p_draw, pred.p_away))
        outcomes.append(
            outcome_from_score(int(match["home_goals"]), int(match["away_goals"]))
        )
        predicted_scores.append(pred.most_likely_score)
        actual_scores.append((int(match["home_goals"]), int(match["away_goals"])))

    if verbose:
        print(f"done", flush=True)
    metrics = summarize(predictions, outcomes, predicted_scores, actual_scores)
    metrics["year"] = year
    metrics["strategy"] = strategy
    metrics["skipped"] = skipped
    return metrics


def run_comparison() -> pd.DataFrame:
    """Compara baseline vs elo_weighted en los 3 Mundiales."""
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")

    print("Cargando/precomputando timeline de Elo...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    print(f"Timeline listo: {len(timeline)} fechas\n", flush=True)

    df = load_martj42_csv(csv_path)
    rows = []
    for year in [2014, 2018, 2022]:
        print(f"=== WC {year} ===", flush=True)
        for strategy in ["baseline", "elo_weighted"]:
            print(f"  {strategy}:", end=" ", flush=True)
            m = backtest_strategy(df, year, timeline, strategy=strategy)
            n = m.get("n", 0)
            print(
                f"n={n}, brier={m.get('brier', 0):.3f}, "
                f"sign={m.get('sign_accuracy', 0):.1%}, "
                f"exact={m.get('exact_score_accuracy', 0):.1%}",
                flush=True,
            )
            rows.append(m)

    df_res = pd.DataFrame(rows)
    print()
    print("=" * 80)
    print("COMPARATIVA")
    print("=" * 80)
    cols = ["year", "strategy", "n", "brier", "rps", "log_loss", "sign_accuracy", "exact_score_accuracy"]
    print(df_res[cols].to_string(index=False))

    print()
    print("=" * 80)
    print("PROMEDIOS POR ESTRATEGIA")
    print("=" * 80)
    avg = df_res.groupby("strategy")[
        ["brier", "rps", "log_loss", "sign_accuracy", "exact_score_accuracy"]
    ].mean()
    print(avg.round(4).to_string())
    return df_res


if __name__ == "__main__":
    run_comparison()
