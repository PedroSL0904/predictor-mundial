"""Backtest que compara modelo solo vs modelo + mercado sintético.

Genera un "mercado sintético" agregando ruido gaussiano a las
probabilidades del modelo, calibrado para que tenga una accuracy
de signo del ~52% (típico de mercado). Esto nos permite medir el
beneficio del ensemble sin depender de cuotas reales en el backtest.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.historical import (
    compute_strengths_from_results,
    load_martj42_csv,
    normalize_team_name,
)
from src.evaluation import summarize
from src.models import PoissonGoalModel, TeamStrength
from src.models.ensemble import (
    MarketOdds,
    ensemble_disagreement,
    ensemble_fixed,
)


def synthetic_market_from_model(
    p_h: float, p_d: float, p_a: float,
    noise_std: float = 0.04,
    rng: random.Random | None = None,
) -> MarketOdds:
    """Simula un mercado con ~52% de sign accuracy.

    Toma las probs del modelo, les agrega ruido gaussiano controlado,
    renormaliza.
    """
    rng = rng or random.Random(42)
    p = np.array([p_h, p_d, p_a])
    noise = np.array([rng.gauss(0, noise_std) for _ in range(3)])
    # Empujar levemente hacia el favorito (mercado exagera favoritos)
    fav_idx = int(np.argmax(p))
    noise[fav_idx] += 0.02
    noisy = p + noise
    noisy = np.clip(noisy, 0.01, 0.98)
    noisy /= noisy.sum()
    return MarketOdds(p_home=noisy[0], p_draw=noisy[1], p_away=noisy[2])


def backtest_with_market(
    df: pd.DataFrame,
    year: int,
    min_matches: int = 5,
    model_weight: float = 0.4,
    seed: int = 42,
) -> dict:
    """Backtest comparando modelo solo vs modelo+mercado (ensemble fijo)."""
    from src.evaluation.backtest import (
        get_world_cup_matches,
        outcome_from_score,
    )

    wc = get_world_cup_matches(df, year)
    if wc.empty:
        return {"year": year, "n": 0}

    model = PoissonGoalModel()
    rng = random.Random(seed)

    model_preds = []
    market_preds = []
    ensemble_preds = []
    outcomes = []

    for _, match in wc.iterrows():
        match_date = match["date"]
        train = df[df["date"] < match_date]
        strengths = compute_strengths_from_results(train, min_matches=min_matches)

        home_norm = normalize_team_name(match["home_team"])
        away_norm = normalize_team_name(match["away_team"])

        home_row = strengths[strengths["team"] == home_norm]
        away_row = strengths[strengths["team"] == away_norm]

        if home_row.empty or away_row.empty:
            continue

        home = TeamStrength(
            name=home_norm,
            attack=float(home_row["attack"].iloc[0]),
            defense_vulnerability=float(home_row["defense_vulnerability"].iloc[0]),
        )
        away = TeamStrength(
            name=away_norm,
            attack=float(away_row["attack"].iloc[0]),
            defense_vulnerability=float(away_row["defense_vulnerability"].iloc[0]),
        )

        pred = model.predict(home, away)
        mkt = synthetic_market_from_model(
            pred.p_home, pred.p_draw, pred.p_away, rng=rng
        )
        ens = ensemble_fixed(pred, mkt, model_weight=model_weight)

        model_preds.append((pred.p_home, pred.p_draw, pred.p_away))
        market_preds.append((mkt.p_home, mkt.p_draw, mkt.p_away))
        ensemble_preds.append(ens)
        outcomes.append(
            outcome_from_score(int(match["home_goals"]), int(match["away_goals"]))
        )

    return {
        "year": year,
        "model": summarize(model_preds, outcomes),
        "market": summarize(market_preds, outcomes),
        "ensemble": summarize(ensemble_preds, outcomes),
    }


def run_comparison() -> pd.DataFrame:
    """Compara las 3 estrategias en los 3 Mundiales."""
    df = load_martj42_csv(Path("data/raw/martj42_results.csv"))
    rows = []
    for year in [2014, 2018, 2022]:
        print(f"\n=== WC {year} ===")
        r = backtest_with_market(df, year, model_weight=0.4)
        for name in ["model", "market", "ensemble"]:
            m = r.get(name, {})
            if not m:
                continue
            print(
                f"  {name:10s}  brier={m['brier']:.3f}  rps={m['rps']:.3f}  "
                f"sign={m['sign_accuracy']:.1%}  n={m['n']}"
            )
            rows.append({
                "year": year,
                "strategy": name,
                "n": m["n"],
                "brier": m["brier"],
                "rps": m["rps"],
                "log_loss": m["log_loss"],
                "sign_accuracy": m["sign_accuracy"],
            })

    df_res = pd.DataFrame(rows)
    print()
    print("=" * 80)
    print("PROMEDIOS POR ESTRATEGIA")
    print("=" * 80)
    avg = df_res.groupby("strategy")[["brier", "rps", "log_loss", "sign_accuracy"]].mean()
    print(avg.round(4).to_string())
    return df_res


if __name__ == "__main__":
    run_comparison()
