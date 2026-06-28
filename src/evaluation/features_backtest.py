"""Backtest comparativo: con vs sin features historicas (H2H, momentum, WC).

Para cada mundial (2014, 2018, 2022):
- Baseline: TournamentSimulator sin features
- Con features: TournamentSimulator con features
- Compara Brier, log loss, RPS, sign accuracy

Output: tabla Markdown en data/processed/features_backtest.md
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv, normalize_team_name
from src.domain import outcome_from_score
from src.evaluation.backtest_cached import get_world_cup_matches
from src.features.strengths_cache import StrengthsCache
from src.simulation.wc2026_simulate import TournamentSimulator


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcomes] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.log(np.maximum(probs[np.arange(len(probs)), outcomes], eps)).mean())


def _sign_acc(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((np.argmax(probs, axis=1) == outcomes).mean())


def _rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    cum_probs = np.cumsum(probs, axis=1)
    cum_outs = np.zeros_like(cum_probs)
    cum_outs[np.arange(len(outcomes)), outcomes] = 1
    cum_outs = np.cumsum(cum_outs, axis=1)
    return float(((cum_probs - cum_outs) ** 2).sum(axis=1).mean() / 2)


def _per_metrics(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    return {
        "brier": _brier(probs, outcomes),
        "log_loss": _log_loss(probs, outcomes),
        "sign_acc": _sign_acc(probs, outcomes),
        "rps": _rps(probs, outcomes),
        "n": int(len(outcomes)),
    }


def backtest_year(
    df: pd.DataFrame,
    timeline: dict,
    cache: StrengthsCache,
    year: int,
    enable_features: bool,
) -> dict:
    """Corre el backtest para un mundial con/sin features."""
    sim = TournamentSimulator(
        df, timeline, cache,
        league_avg_multiplier=1.0,  # backtest usa datos historicos
        as_of=f"{year}-01-01",
        calibrator=None,
        injuries=None,
        enable_historical_features=enable_features,
    )

    wc = get_world_cup_matches(df, year)
    if wc.empty:
        return {"year": year, "n": 0}

    wc_sorted = wc.sort_values("date").reset_index(drop=True)
    first_date = str(wc_sorted["date"].iloc[0])[:10]
    sim.as_of = first_date
    sim.cache.set_elo_snapshot(first_date)
    sim.strengths = sim.cache.get_strengths(first_date)
    sim.strength_by_team = sim.strengths.set_index("team")
    sim.elo_lookup = {team: elo for team, elo in sim.elo_lookup.items()}

    probs_list: list[list[float]] = []
    outcomes_list: list[int] = []
    for _, m in wc_sorted.iterrows():
        home = normalize_team_name(m["home_team"])
        away = normalize_team_name(m["away_team"])
        try:
            p = sim.predict(home, away)
            probs_list.append([p.p_home, p.p_draw, p.p_away])
        except Exception:
            continue
        outs = outcome_from_score(int(m["home_goals"]), int(m["away_goals"]))
        outcomes_list.append({"H": 0, "D": 1, "A": 2}[outs.value])
    if not probs_list:
        return {"year": year, "n": 0}
    probs = np.array(probs_list)
    outcomes = np.array(outcomes_list)
    m = _per_metrics(probs, outcomes)
    m["year"] = year
    return m


def main() -> None:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.parquet")
    print("Cargando datos...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    years = (2014, 2018, 2022)
    cache = StrengthsCache(df, timeline)
    rows: list[dict] = []
    for y in years:
        print(f"=== WC {y} ===", flush=True)
        t0 = time.time()
        m_off = backtest_year(df, timeline, cache, y, enable_features=False)
        m_on = backtest_year(df, timeline, cache, y, enable_features=True)
        elapsed = time.time() - t0
        print(f"  OFF: brier={m_off.get('brier', 0):.4f} sign={m_off.get('sign_acc', 0):.1%} "
              f"n={m_off.get('n', 0)}", flush=True)
        print(f"  ON:  brier={m_on.get('brier', 0):.4f} sign={m_on.get('sign_acc', 0):.1%} "
              f"n={m_on.get('n', 0)}", flush=True)
        rows.append({"year": y, "off": m_off, "on": m_on, "elapsed": elapsed})

    # Formato Markdown
    lines: list[str] = []
    lines.append("# Backtest: features historicas (H2H, momentum, WC)\n")
    lines.append("Comparacion con/sin features en WC 2014+2018+2022 (sin lesiones, sin calibrador).\n")
    lines.append("")
    lines.append("| WC | n | Brier OFF | Brier ON | Delta Brier | Sign OFF | Sign ON | LogLoss OFF | LogLoss ON |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        off, on = r["off"], r["on"]
        delta = on.get("brier", 0) - off.get("brier", 0)
        lines.append(
            f"| {r['year']} | {off.get('n', 0)} | "
            f"{off.get('brier', 0):.4f} | {on.get('brier', 0):.4f} | {delta:+.4f} | "
            f"{off.get('sign_acc', 0):.1%} | {on.get('sign_acc', 0):.1%} | "
            f"{off.get('log_loss', 0):.4f} | {on.get('log_loss', 0):.4f} |"
        )
    if rows:
        avg_off = np.mean([r["off"].get("brier", 0) for r in rows])
        avg_on = np.mean([r["on"].get("brier", 0) for r in rows])
        improvement = (avg_off - avg_on) / avg_off * 100
        lines.append("")
        lines.append(f"**Promedios:** OFF={avg_off:.4f}, ON={avg_on:.4f}")
        lines.append(f"Mejora con features: {improvement:+.2f}%")

    md = "\n".join(lines)
    print()
    print(md.encode("ascii", "replace").decode("ascii"), flush=True)
    out_path = Path("data/processed/features_backtest.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"\nReporte guardado en {out_path}")


if __name__ == "__main__":
    main()
