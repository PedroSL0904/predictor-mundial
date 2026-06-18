"""Grid search: optimiza recent_form_n_matches x recent_form_weight.

LOO sobre 5 mundiales (2010, 2014, 2018, 2022; usamos 2014/2018/2022 ya que
2010 entra como LOO mientras se optimiza con los otros 3).

Por cada config corre 3 mundiales (~3 * 90s = 4.5 min por config).
Grid chico para mantener runtime manejable.
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

    print("Cargando/precomputando timeline de Elo...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    print(f"Timeline listo: {len(timeline)} fechas\n", flush=True)

    df = load_martj42_csv(csv_path)
    years = [2014, 2018, 2022]

    # Grid compacto: 4 x 4 = 16 configs
    n_matches_grid = [3, 5]
    weight_grid = [0.0, 0.2]

    rows = []
    t0 = time.time()
    total = len(n_matches_grid) * len(weight_grid)
    idx = 0
    for n_m in n_matches_grid:
        for w in weight_grid:
            idx += 1
            cfg_t0 = time.time()
            avg_brier = 0.0
            avg_rps = 0.0
            avg_log = 0.0
            avg_sign = 0.0
            n_total = 0
            for year in years:
                m = backtest_strategy(
                    df, year, timeline,
                    strategy="elo_weighted",
                    recent_form_n_matches=n_m,
                    recent_form_weight=w,
                    verbose=False,
                )
                avg_brier += m.get("brier", 0)
                avg_rps += m.get("rps", 0)
                avg_log += m.get("log_loss", 0)
                avg_sign += m.get("sign_accuracy", 0)
                n_total += m.get("n", 0)
            n_years = len(years)
            avg_brier /= n_years
            avg_rps /= n_years
            avg_log /= n_years
            avg_sign /= n_years
            elapsed = time.time() - cfg_t0
            rows.append({
                "n_matches": n_m,
                "weight_recent": w,
                "brier": round(avg_brier, 4),
                "rps": round(avg_rps, 4),
                "log_loss": round(avg_log, 4),
                "sign": round(avg_sign, 4),
                "n_total": n_total,
                "sec": round(elapsed, 1),
            })
            print(
                f"[{idx}/{total}] n={n_m} w={w:.1f} "
                f"brier={avg_brier:.4f} sign={avg_sign:.4f} "
                f"({elapsed:.0f}s, total {time.time()-t0:.0f}s)",
                flush=True,
            )

    res = pd.DataFrame(rows).sort_values("brier")
    print("\n" + "=" * 80)
    print("RANKING POR BRIER")
    print("=" * 80)
    print(res.to_string(index=False))
    res.to_csv("grid_recent_form_results.csv", index=False)
    print("\nGuardado en grid_recent_form_results.csv")


if __name__ == "__main__":
    main()
