"""Grid v5: usa compute_weighted_strengths (la implementacion real, no
la vectorizada con bug). Cross-validation LOO por mundial.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv, normalize_team_name
from src.evaluation.backtest import get_world_cup_matches
from src.evaluation.backtest_elo import get_elo_at
from src.evaluation.metrics import brier_score, ranked_probability_score, log_loss
from src.features.strengths import compute_weighted_strengths
from src.models import PoissonGoalModel, TeamStrength


PARAM_SPACE = {
    "elo_sigma": [100, 150, 200, 250, 300],
    "recency_half_life_days": [540, 730, 1000, 1500, 2000],
    "shrinkage_matches": [5, 8, 10, 15, 20, 30],
    "min_weighted_matches": [5, 8, 10],
    "draw_penalty_threshold": [0.04, 0.06, 0.08, 0.10],
    "draw_penalty_strength": [0.0, 0.05, 0.10, 0.15, 0.20],
    "elo_gap_inflation": [0.0, 0.04, 0.08, 0.15, 0.20, 0.30],
    "draw_boost": [0.0, 0.10, 0.20, 0.25, 0.30],
}


def evaluate_year(df, year, params, timeline):
    wc = get_world_cup_matches(df, year)
    if wc.empty:
        return None
    model = PoissonGoalModel(
        draw_penalty_threshold=params["draw_penalty_threshold"],
        draw_penalty_strength=params["draw_penalty_strength"],
        elo_gap_inflation=params["elo_gap_inflation"],
        draw_boost=params.get("draw_boost", 0.0),
    )
    preds, outs, pred_scores, actual_scores = [], [], [], []
    for _, m in wc.iterrows():
        match_date = str(m["date"])[:10]
        train = df[df["date"] < m["date"]].copy()
        if train.empty:
            continue
        elo_lookup = get_elo_at(timeline, match_date)
        hn = normalize_team_name(m["home_team"])
        an = normalize_team_name(m["away_team"])
        s = compute_weighted_strengths(
            train, elo_lookup=elo_lookup,
            elo_sigma=params["elo_sigma"],
            recency_half_life_days=params["recency_half_life_days"],
            shrinkage_matches=params["shrinkage_matches"],
            min_weighted_matches=params["min_weighted_matches"],
        )
        hr = s[s["team"] == hn]
        ar = s[s["team"] == an]
        if hr.empty or ar.empty:
            continue
        h = TeamStrength(
            name=hn, attack=float(hr["attack"].iloc[0]),
            defense_vulnerability=float(hr["defense_vulnerability"].iloc[0]),
        )
        a = TeamStrength(
            name=an, attack=float(ar["attack"].iloc[0]),
            defense_vulnerability=float(ar["defense_vulnerability"].iloc[0]),
        )
        home_elo = elo_lookup.get(hn, 1500.0)
        away_elo = elo_lookup.get(an, 1500.0)
        pred = model.predict(h, a, home_elo=home_elo, away_elo=away_elo)
        preds.append((pred.p_home, pred.p_draw, pred.p_away))
        actual = "H" if m["home_goals"] > m["away_goals"] else ("A" if m["home_goals"] < m["away_goals"] else "D")
        outs.append(actual)
        pred_scores.append(pred.most_likely_score)
        actual_scores.append((int(m["home_goals"]), int(m["away_goals"])))
    if not preds:
        return None
    n = len(preds)
    brier = sum(brier_score(p, o) for p, o in zip(preds, outs)) / n
    rps = sum(ranked_probability_score(p, o) for p, o in zip(preds, outs)) / n
    ll = sum(log_loss(p, o) for p, o in zip(preds, outs)) / n
    sign = sum(1 for p, o in zip(preds, outs) if np.argmax(p) == ["H","D","A"].index(o)) / n
    return {"brier": brier, "rps": rps, "log_loss": ll, "sign_accuracy": sign, "n": n}


if __name__ == "__main__":
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    years = [2006, 2010, 2014, 2018, 2022]
    print(f"Años a evaluar: {years}\n")

    rng = np.random.default_rng(42)
    n_samples = 200
    samples = []
    for _ in range(n_samples):
        p = {}
        for k, v in PARAM_SPACE.items():
            val = rng.choice(v)
            if k in ("shrinkage_matches", "min_weighted_matches"):
                p[k] = int(val)
            else:
                p[k] = float(val)
        samples.append(p)

    results = []
    t0 = time.time()
    for i, params in enumerate(samples):
        per_year = {}
        for y in years:
            m = evaluate_year(df, y, params, timeline)
            if m is None:
                continue
            per_year[y] = m
        if len(per_year) < 5:
            continue
        avg_brier = np.mean([per_year[y]["brier"] for y in years])
        avg_sign = np.mean([per_year[y]["sign_accuracy"] for y in years])
        results.append({
            "params": params,
            "avg_brier": avg_brier,
            "avg_sign": avg_sign,
            "per_year": {str(y): per_year[y] for y in years},
        })
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            best = min((r["avg_brier"] for r in results), default=1.0)
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(f"  [{i+1}/{len(samples)}] elapsed={elapsed:.0f}s eta={eta:.0f}s best_avg_brier={best:.4f}", flush=True)

    # Ranking
    results.sort(key=lambda x: x["avg_brier"])
    print()
    print("=" * 100)
    print(f"TOP 20 POR AVG_BRIER (5 mundial LOO)")
    print("=" * 100)
    print(f"{'avg_brier':>9} {'avg_sign':>8} | per-year briers")
    for r in results[:20]:
        p = r["params"]
        per_y = " ".join(f"{r['per_year'][str(y)]['brier']:.3f}" for y in years)
        print(f"{r['avg_brier']:>9.4f} {r['avg_sign']*100:>7.1f}% | {per_y}")
        print(f"           | sigma={p['elo_sigma']:.0f} rec={p['recency_half_life_days']:.0f} "
              f"shr={p['shrinkage_matches']} mwm={p['min_weighted_matches']} "
              f"dthr={p['draw_penalty_threshold']:.2f} dstr={p['draw_penalty_strength']:.2f} "
              f"einf={p['elo_gap_inflation']:.2f} boost={p['draw_boost']:.2f}")

    out = Path("data/processed/grid_search_v5.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nGuardado en {out}")
