"""Mini grid: solo 4 configs prometedoras para validar en 1h.

Configs:
  1. baseline actual (rf_w=0.2, db=0.20, disp=0.0)  [replicar]
  2. rf_w=0.2, db=0.30, disp=0.0  (mas boost a draws)
  3. rf_w=0.2, db=0.10, disp=0.0  (menos boost)
  4. rf_w=0.2, db=0.20, disp=0.10  (negative binomial)

LOO: 2014+2018+2022. 4 configs * 3 WC = 12 backtests. ~26 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.evaluation.backtest_elo import backtest_strategy


def main() -> None:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")
    print("Cargando timeline...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    years = [2014, 2018, 2022]

    configs = [
        {"name": "rf02_db020_disp0", "rf_w": 0.20, "db": 0.20, "disp": 0.0},
        {"name": "rf02_db030_disp0", "rf_w": 0.20, "db": 0.30, "disp": 0.0},
        {"name": "rf02_db010_disp0", "rf_w": 0.20, "db": 0.10, "disp": 0.0},
        {"name": "rf02_db020_disp10", "rf_w": 0.20, "db": 0.20, "disp": 0.10},
    ]

    rows = []
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        cfg_t0 = time.time()
        briers = []
        signs = []
        for year in years:
            m = backtest_strategy(
                df, year, timeline,
                strategy="elo_weighted",
                recent_form_n_matches=5,
                recent_form_weight=cfg["rf_w"],
                draw_boost=cfg["db"],
                dispersion=cfg["disp"],
                verbose=False,
            )
            briers.append(m.get("brier", 0))
            signs.append(m.get("sign_accuracy", 0))
        avg_brier = sum(briers) / len(briers)
        avg_sign = sum(signs) / len(signs)
        elapsed = time.time() - cfg_t0
        rows.append({
            "config": cfg["name"],
            "brier_2014": round(briers[0], 4),
            "brier_2018": round(briers[1], 4),
            "brier_2022": round(briers[2], 4),
            "avg_brier": round(avg_brier, 4),
            "avg_sign": round(avg_sign, 4),
            "sec": round(elapsed, 1),
        })
        print(
            f"[{i}/{len(configs)}] {cfg['name']} "
            f"brier_2014={briers[0]:.4f} 2018={briers[1]:.4f} 2022={briers[2]:.4f} "
            f"avg={avg_brier:.4f} ({elapsed:.0f}s, total {(time.time()-t0)/60:.1f}m)",
            flush=True,
        )

    res = pd.DataFrame(rows).sort_values("avg_brier")
    print("\n" + "=" * 80)
    print("RANKING POR BRIER PROMEDIO")
    print("=" * 80)
    print(res.to_string(index=False))
    res.to_csv("grid_mini_results.csv", index=False)


if __name__ == "__main__":
    main()
