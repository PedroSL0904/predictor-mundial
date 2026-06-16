"""Backtest del modelo Poisson sobre Mundiales pasados.

Evalúa Brier, RPS, log loss y accuracy de signo sobre los partidos ya
jugados de Mundiales 2014, 2018 y 2022.

Usa los strengths calculados **solo con partidos previos** al Mundial
(rolling origin) para evitar leakage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.historical import (
    compute_strengths_from_results,
    load_martj42_csv,
    normalize_team_name,
)
from src.evaluation import summarize
from src.models import PoissonGoalModel, TeamStrength


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


def predict_match(
    home_name: str,
    away_name: str,
    home_strength: TeamStrength,
    away_strength: TeamStrength,
    home_elo: float = 1500.0,
    away_elo: float = 1500.0,
) -> tuple[float, float, float]:
    """Genera (p_home, p_draw, p_away) para un partido."""
    model = PoissonGoalModel()
    pred = model.predict(home_strength, away_strength, home_elo, away_elo)
    return pred.p_home, pred.p_draw, pred.p_away


def outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def backtest_world_cup(
    df: pd.DataFrame,
    year: int,
    min_matches: int = 5,
) -> dict:
    """Backtest sobre un Mundial específico.

    Para cada partido del Mundial:
    1. Calcula strengths usando SOLO partidos previos a ese encuentro.
    2. Predice con el modelo.
    3. Compara con resultado real.
    """
    wc = get_world_cup_matches(df, year)
    if wc.empty:
        return {"year": year, "n": 0}

    predictions = []
    outcomes = []
    predicted_scores = []
    actual_scores = []

    model = PoissonGoalModel()
    skipped = 0

    for _, match in wc.iterrows():
        match_date = match["date"]
        # Training set: todo antes de este partido
        train = df[df["date"] < match_date]
        strengths = compute_strengths_from_results(train, min_matches=min_matches)

        home_norm = normalize_team_name(match["home_team"])
        away_norm = normalize_team_name(match["away_team"])

        home_row = strengths[strengths["team"] == home_norm]
        away_row = strengths[strengths["team"] == away_norm]

        if home_row.empty or away_row.empty:
            skipped += 1
            continue

        home = TeamStrength(
            name=home_norm,
            attack=float(home_row["attack"].iloc[0]),
            defense_vulnerability=float(home_row["defense_vulnerability"].iloc[0]),
            matches=int(home_row["matches"].iloc[0]),
        )
        away = TeamStrength(
            name=away_norm,
            attack=float(away_row["attack"].iloc[0]),
            defense_vulnerability=float(away_row["defense_vulnerability"].iloc[0]),
            matches=int(away_row["matches"].iloc[0]),
        )

        pred = model.predict(home, away)
        predictions.append((pred.p_home, pred.p_draw, pred.p_away))
        outcomes.append(outcome_from_score(int(match["home_goals"]), int(match["away_goals"])))
        predicted_scores.append(pred.most_likely_score)
        actual_scores.append((int(match["home_goals"]), int(match["away_goals"])))

    metrics = summarize(predictions, outcomes, predicted_scores, actual_scores)
    metrics["year"] = year
    metrics["skipped"] = skipped
    return metrics


def backtest_all() -> pd.DataFrame:
    """Backtest sobre Mundiales 2014, 2018, 2022."""
    csv_path = Path("data/raw/martj42_results.csv")
    df = load_martj42_csv(csv_path)

    results = []
    for year in [2014, 2018, 2022]:
        print(f"Backtesting WC {year}...")
        m = backtest_world_cup(df, year)
        results.append(m)
        if m.get("n", 0) > 0:
            print(
                f"  n={m['n']}, brier={m['brier']:.3f}, rps={m['rps']:.3f}, "
                f"log_loss={m['log_loss']:.3f}, sign_acc={m['sign_accuracy']:.1%}, "
                f"exact_score={m.get('exact_score_accuracy', 0):.1%}"
            )
        else:
            print("  (sin partidos)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = backtest_all()
    print()
    print("=" * 70)
    print("RESUMEN BACKTEST")
    print("=" * 70)
    cols = [
        "year", "n", "brier", "rps", "log_loss", "sign_accuracy", "exact_score_accuracy",
        "p_home_avg", "p_draw_avg", "p_away_avg",
    ]
    print(results[cols].to_string(index=False))
    print()
    print("Promedios:")
    means = results[cols].mean(numeric_only=True)
    print(f"  Brier:      {means['brier']:.4f}")
    print(f"  RPS:        {means['rps']:.4f}")
    print(f"  Log loss:   {means['log_loss']:.4f}")
    print(f"  Sign acc:   {means['sign_accuracy']:.1%}")
    print(f"  Exact score: {means['exact_score_accuracy']:.1%}")
