"""Reporte comparativo: Ensemble vs modelos individuales en WC 2014+2018+2022.

Para cada mundial:
- Brier score del ensemble (con pesos LOO)
- Brier score de Poisson, BP, Skellam por separado
- Sign accuracy, log loss, RPS

Output: tabla Markdown + CSV en data/processed/.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.evaluation.ensemble_optimization import (
    EnsembleWeights,
    _brier,
    collect_per_model_predictions,
    loo_optimize_ensemble,
)
from src.features.strengths_cache import StrengthsCache
from src.logging_config import get_logger

logger = get_logger(__name__)


def _sign_acc(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((np.argmax(probs, axis=1) == outcomes).mean())


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.log(np.maximum(probs[np.arange(len(probs)), outcomes], eps)).mean())


def _rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score para 3 outcomes."""
    cum_probs = np.cumsum(probs, axis=1)
    cum_outs = np.zeros_like(cum_probs)
    cum_outs[np.arange(len(outcomes)), outcomes] = 1
    cum_outs = np.cumsum(cum_outs, axis=1)
    return float(((cum_probs - cum_outs) ** 2).sum(axis=1).mean() / 2)


def _per_model_metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "brier": _brier(probs, outcomes),
        "log_loss": _log_loss(probs, outcomes),
        "rps": _rps(probs, outcomes),
        "sign_acc": _sign_acc(probs, outcomes),
        "n": int(len(outcomes)),
    }


def build_comparison_report(
    df,
    cache: StrengthsCache,
    timeline: dict,
    weights: EnsembleWeights,
    years: tuple[int, ...] = (2014, 2018, 2022),
) -> dict:
    """Para cada mundial, evalua ensemble + 3 modelos individuales.

    Returns:
        dict con tabla formateada + datos crudos.
    """
    rows: list[dict] = []
    raw_per_year: dict[int, dict] = {}
    for y in years:
        t0 = time.time()
        probs_p, probs_bp, probs_sk, outs = collect_per_model_predictions(
            df, y, cache, timeline, verbose=False
        )
        if len(outs) == 0:
            continue
        ens = (
            weights.poisson * probs_p
            + weights.bivariate_poisson * probs_bp
            + weights.skellam * probs_sk
        )
        raw_per_year[y] = {
            "poisson": probs_p,
            "bp": probs_bp,
            "skellam": probs_sk,
            "ensemble": ens,
            "outcomes": outs,
        }
        m_p = _per_model_metrics(probs_p, outs)
        m_bp = _per_model_metrics(probs_bp, outs)
        m_s = _per_model_metrics(probs_sk, outs)
        m_e = _per_model_metrics(ens, outs)
        rows.append(
            {
                "year": y,
                "n": m_p["n"],
                "brier_ensemble": m_e["brier"],
                "brier_poisson": m_p["brier"],
                "brier_bp": m_bp["brier"],
                "brier_skellam": m_s["brier"],
                "log_loss_ensemble": m_e["log_loss"],
                "sign_acc_ensemble": m_e["sign_acc"],
                "sign_acc_poisson": m_p["sign_acc"],
                "elapsed": time.time() - t0,
            }
        )

    return {
        "rows": rows,
        "raw": raw_per_year,
        "weights": weights,
    }


def format_markdown(report: dict) -> str:
    """Formatea el reporte como Markdown."""
    weights: EnsembleWeights = report["weights"]
    rows = report["rows"]
    lines: list[str] = []
    lines.append("# Ensemble vs modelos individuales en WC 2014+2018+2022\n")
    lines.append(f"**Pesos (LOO):** P={weights.poisson:.2f}, "
                 f"BP={weights.bivariate_poisson:.2f}, "
                 f"S={weights.skellam:.2f}\n")
    lines.append(f"**Brier promedio en train (LOO):** {weights.brier_train:.4f}\n")
    lines.append("")
    lines.append("| WC | n | Brier Ens | Brier P | Brier BP | Brier S | Sign Ens | Sign P |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['year']} | {r['n']} | {r['brier_ensemble']:.4f} | "
            f"{r['brier_poisson']:.4f} | {r['brier_bp']:.4f} | "
            f"{r['brier_skellam']:.4f} | {r['sign_acc_ensemble']:.1%} | "
            f"{r['sign_acc_poisson']:.1%} |"
        )
    lines.append("")
    if rows:
        avg_e = np.mean([r["brier_ensemble"] for r in rows])
        avg_p = np.mean([r["brier_poisson"] for r in rows])
        avg_bp = np.mean([r["brier_bp"] for r in rows])
        avg_s = np.mean([r["brier_skellam"] for r in rows])
        improvement = (avg_p - avg_e) / avg_p * 100
        lines.append(f"**Promedios:** Ens={avg_e:.4f}, P={avg_p:.4f}, "
                     f"BP={avg_bp:.4f}, S={avg_s:.4f}")
        lines.append(f"Mejora ensemble vs Poisson: {improvement:+.2f}%")
    return "\n".join(lines)


def main() -> None:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.parquet")
    logger.info("Cargando datos...")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)

    logger.info("\n=== LOO 3 mundial ===")
    t0 = time.time()
    weights = loo_optimize_ensemble(df, cache, timeline)
    logger.info(f"Pesos finales: {weights}")
    logger.info(f"Tiempo: {time.time()-t0:.1f}s\n")

    logger.info("=== Comparativa Ensemble vs individuales ===")
    report = build_comparison_report(df, cache, timeline, weights)
    md = format_markdown(report)
    logger.info(md)

    out_path = Path("data/processed/ensemble_comparison.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    logger.info(f"\nReporte guardado en {out_path}")


if __name__ == "__main__":
    main()
