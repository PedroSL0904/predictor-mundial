"""Verificacion INDEPENDIENTE de las metricas del WC 2026.

Recomputa brier, sign accuracy, log loss desde wc2026_predictions.csv
SIN depender de compute_metrics() del CLI. Compara con los valores
publicados en WC2026_README.md.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("wc2026_predictions.csv")
README_PATH = Path("WC2026_README.md")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Total partidos: {len(df)}")
    logger.info(f"Columnas: {list(df.columns)}")

    # Solo partidos FT (jugados)
    played = df[df["played"] == True].copy()  # noqa: E712
    logger.info(f"Partidos FT: {len(played)}")

    # Outcome real: H si home_score > away_score, D si =, A si <
    def outcome(r: pd.Series) -> int:
        if r["home_score"] > r["away_score"]:
            return 0  # H
        if r["home_score"] < r["away_score"]:
            return 2  # A
        return 1  # D

    played = played.copy()
    played["outcome"] = played.apply(outcome, axis=1)
    played["pick"] = played[["p_h", "p_d", "p_a"]].values.argmax(axis=1)

    # 1. Brier score
    probs = played[["p_h", "p_d", "p_a"]].values
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(played)), played["outcome"].values] = 1
    brier = ((probs - onehot) ** 2).sum(axis=1).mean()
    logger.info(f"\nBrier score: {brier:.4f}")

    # 2. Log loss
    eps = 1e-9
    selected = probs[np.arange(len(played)), played["outcome"].values]
    logloss = -np.log(np.maximum(selected, eps)).mean()
    logger.info(f"Log loss: {logloss:.4f}")

    # 3. Sign accuracy
    sign_acc = (played["pick"].values == played["outcome"].values).mean()
    logger.info(f"Sign accuracy: {sign_acc:.1%}")

    # 4. Desglose por outcome
    logger.info("\nDesglose por outcome real:")
    for o, name in [(0, "Home (H)"), (1, "Draw (D)"), (2, "Away (A)")]:
        n = (played["outcome"] == o).sum()
        pct = n / len(played) * 100
        avg_p = probs[played["outcome"].values == o, o].mean()
        logger.info(f"  {name}: {n} partidos ({pct:.0f}%), avg P={avg_p:.1%}")

    # 5. Comparar con lo que dice el README
    logger.info("\n--- Comparacion con WC2026_README.md ---")
    readme = README_PATH.read_text(encoding="utf-8")
    brier_match = re.search(r"Brier score \(1X2\)\s*\|\s*\*\*([\d.]+)\*\*", readme)
    sign_match = re.search(r"Sign accuracy\s*\|\s*\*\*([\d.]+)%\*\*", readme)
    logloss_match = re.search(r"Log loss\s*\|\s*\*\*([\d.]+)\*\*", readme)

    if brier_match:
        readme_brier = float(brier_match.group(1))
        match_str = "OK" if abs(readme_brier - brier) < 0.001 else "MISMATCH"
        logger.info(f"  Brier README: {readme_brier:.4f} | Computed: {brier:.4f} | {match_str}")
    if sign_match:
        readme_sign = float(sign_match.group(1))
        match_str = "OK" if abs(readme_sign - sign_acc * 100) < 0.5 else "MISMATCH"
        logger.info(f"  Sign README: {readme_sign:.1f}% | Computed: {sign_acc:.1%} | {match_str}")
    if logloss_match:
        readme_logloss = float(logloss_match.group(1))
        match_str = "OK" if abs(readme_logloss - logloss) < 0.005 else "MISMATCH"
        logger.info(f"  LogLoss README: {readme_logloss:.4f} | Computed: {logloss:.4f} | {match_str}")

    # 6. Predicciones top por partido FT (las mas extremas)
    logger.info("\n--- Top 5 picks mas acertados ---")
    played["correct"] = played["pick"] == played["outcome"]
    top5 = played.nlargest(5, "p_h")  # los que mas se jugo el local
    for _, r in top5.iterrows():
        logger.info(
            f"  {r['home']} vs {r['away']}: pred H {r['p_h']:.0%}, "
            f"real {int(r['home_score'])}-{int(r['away_score'])} "
            f"({'OK' if r['correct'] else 'X'})"
        )

    # 7. Validacion contra baseline
    logger.info("\n--- Comparacion con baselines ---")
    # Baseline: predecir 1/3 cada uno
    uniform_brier = ((1/3 - onehot) ** 2).sum(axis=1).mean()
    logger.info(f"  Uniform (1/3 cada uno): brier = {uniform_brier:.4f}")
    # Baseline: siempre favorito (el mas probable)
    always_fav_brier = ((probs.max(axis=1) - onehot.max(axis=1)) ** 2).sum() / len(played)
    # No es brier sino MSE simple
    improvement_vs_uniform = (uniform_brier - brier) / uniform_brier * 100
    logger.info(f"  Nuestro modelo:         brier = {brier:.4f}")
    logger.info(f"  Mejora vs uniform:     {improvement_vs_uniform:.1f}%")


if __name__ == "__main__":
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    main()
