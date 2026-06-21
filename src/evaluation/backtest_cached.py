"""Backtest que usa StrengthsCache para ser ~80x mas rapido."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import precompute_and_cache
from src.data.historical import (
    compute_strengths_from_results,
    load_martj42_csv,
    normalize_team_name,
)
from src.evaluation.metrics import summarize
from src.features.recent_form import (
    blend_recent_with_historical,
    compute_recent_form,
)
from src.features.strengths_cache import StrengthsCache
from src.models import PoissonGoalModel, TeamStrength

# Constantes y helpers para backtest de Mundiales
WORLD_CUPS = {
    2006: ("2006-06-09", "2006-07-09"),
    2010: ("2010-06-11", "2010-07-11"),
    2014: ("2014-06-12", "2014-07-13"),
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}


def get_world_cup_matches(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filtra partidos del Mundial para un año dado."""
    start, end = WORLD_CUPS[year]
    mask = (
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= start)
        & (df["date"] <= end)
    )
    return df[mask].copy()


def outcome_from_score(home_goals: int, away_goals: int) -> str:
    """Convierte (home_goals, away_goals) a H/D/A."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def get_elo_at(timeline: dict[str, dict[str, float]], as_of: str) -> dict[str, float]:
    candidates = [d for d in timeline if d <= as_of]
    if not candidates:
        return {}
    return timeline[max(candidates)]


def backtest_strategy_cached(
    df: pd.DataFrame,
    year: int,
    timeline: dict[str, dict[str, float]],
    cache: StrengthsCache,
    strategy: str = "elo_weighted",
    min_weighted_matches: float | None = None,
    recent_form_n_matches: int | None = None,
    recent_form_weight: float | None = None,
    draw_boost: float | None = None,
    draw_penalty_threshold: float | None = None,
    dispersion: float | None = None,
    verbose: bool = True,
) -> dict:
    """Backtest rapido usando StrengthsCache precomputado."""
    from src.config import get_settings
    settings = get_settings()
    if min_weighted_matches is None:
        min_weighted_matches = settings.min_weighted_matches
    if recent_form_n_matches is None:
        recent_form_n_matches = settings.recent_form_n_matches
    if recent_form_weight is None:
        recent_form_weight = settings.recent_form_weight

    wc = get_world_cup_matches(df, year)
    if wc.empty:
        return {"year": year, "strategy": strategy, "n": 0}

    wc_sorted = wc.sort_values("date").reset_index(drop=True)
    first_date = str(wc_sorted["date"].iloc[0])[:10]

    # Set elo snapshot una sola vez por Mundial
    cache.set_elo_snapshot(first_date)

    model = PoissonGoalModel(
        draw_penalty_threshold=(
            draw_penalty_threshold
            if draw_penalty_threshold is not None
            else settings.draw_penalty_threshold
        ),
        draw_penalty_strength=settings.draw_penalty_strength,
        elo_gap_inflation=settings.elo_gap_inflation,
        dispersion=(dispersion if dispersion is not None else 0.0),
        draw_boost=(
            draw_boost
            if draw_boost is not None
            else settings.draw_boost
        ),
    )
    predictions = []
    outcomes = []
    predicted_scores = []
    actual_scores = []
    skipped = 0

    total = len(wc_sorted)
    for i, (_, match) in enumerate(wc_sorted.iterrows()):
        if verbose and i % 10 == 0:
            print(f"    [{i}/{total}]", end=" ", flush=True)
        match_date = str(match["date"])[:10]
        home_norm = normalize_team_name(match["home_team"])
        away_norm = normalize_team_name(match["away_team"])

        if strategy == "baseline":
            # baseline no usa cache, calcular como antes
            train = df[df["date"] < match["date"]].copy()
            strengths = compute_strengths_from_results(train, min_matches=5)
        elif strategy == "elo_weighted":
            # Avanzar el cache hasta este partido
            strengths = cache.get_strengths(
                match_date,
                shrinkage_matches=settings.shrinkage_matches,
                min_weighted_matches=min_weighted_matches,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Blend con forma reciente (opcional)
        if recent_form_n_matches > 0 and recent_form_weight > 0:
            train_for_recent = df[df["date"] < match["date"]].copy()
            recent = compute_recent_form(
                train_for_recent,
                as_of=match_date,
                n_matches=recent_form_n_matches,
                min_matches=min(3, recent_form_n_matches),
            )
            strengths = blend_recent_with_historical(
                strengths, recent, weight_recent=recent_form_weight,
            )

        h = strengths[strengths["team"] == home_norm]
        a = strengths[strengths["team"] == away_norm]

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
        print("done", flush=True)
    metrics = summarize(predictions, outcomes, predicted_scores, actual_scores)
    metrics["year"] = year
    metrics["strategy"] = strategy
    metrics["skipped"] = skipped
    return metrics


def run_cached_comparison() -> pd.DataFrame:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")
    print("Cargando/precomputando timeline de Elo...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    print(f"Timeline listo: {len(timeline)} fechas\n", flush=True)

    df = load_martj42_csv(csv_path)

    print("Construyendo StrengthsCache (una vez)...", flush=True)
    import time
    t0 = time.time()
    cache = StrengthsCache(df, timeline)
    print(f"Cache listo en {time.time()-t0:.1f}s\n", flush=True)

    rows = []
    for year in [2014, 2018, 2022]:
        print(f"=== WC {year} ===", flush=True)
        for strategy in ["baseline", "elo_weighted"]:
            print(f"  {strategy}:", end=" ", flush=True)
            m = backtest_strategy_cached(
                df, year, timeline, cache, strategy=strategy
            )
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
    run_cached_comparison()
