"""Validacion de la mejor config (n=8, w=0.5) en WC 2014 y WC 2022."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.evaluation.backtest_elo import backtest_strategy


def main() -> None:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")

    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    # Top configs del grid 2018
    configs = [
        (8, 0.5),   # ganador brier
        (12, 0.5),
        (8, 0.3),
        (12, 0.3),
    ]
    years = [2014, 2022]

    for n_m, w in configs:
        for year in years:
            t0 = time.time()
            m = backtest_strategy(
                df, year, timeline,
                strategy="elo_weighted",
                recent_form_n_matches=n_m,
                recent_form_weight=w,
                verbose=False,
            )
            elapsed = time.time() - t0
            print(
                f"n={n_m} w={w:.2f} year={year} "
                f"brier={m.get('brier',0):.4f} rps={m.get('rps',0):.4f} "
                f"sign={m.get('sign_accuracy',0):.4f} n={m.get('n',0)} ({elapsed:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
