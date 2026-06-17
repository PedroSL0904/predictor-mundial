"""Backtest del ensemble de modelos con progreso en tiempo real."""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv, normalize_team_name
from src.evaluation.backtest import get_world_cup_matches
from src.evaluation.backtest_elo import get_elo_at
from src.evaluation.metrics import brier_score, ranked_probability_score, log_loss
from src.features.strengths import compute_weighted_strengths
from src.models import EnsembleModel, TeamStrength


def evaluate_ensemble(
    df: pd.DataFrame,
    timeline: dict,
    params: dict,
    weight_poisson: float,
    weight_bivariate: float,
    weight_xg: float,
    rho_bivariate: float = 0.05,
    use_xg: bool = False,
) -> dict:
    """Evalúa el ensemble en todos los mundiales (2006-2022)."""
    # Extraer solo parámetros del modelo (no de strengths)
    model_params = {
        "draw_penalty_threshold": params["draw_penalty_threshold"],
        "draw_penalty_strength": params["draw_penalty_strength"],
        "elo_gap_inflation": params["elo_gap_inflation"],
        "draw_boost": params["draw_boost"],
    }
    
    model = EnsembleModel(
        weight_poisson=weight_poisson,
        weight_bivariate=weight_bivariate,
        weight_xg=weight_xg,
        model_params=model_params,
        rho_bivariate=rho_bivariate,
    )

    all_preds = []
    all_outcomes = []
    all_pred_scores = []
    all_actual_scores = []

    for year in [2006, 2010, 2014, 2018, 2022]:
        wc = get_world_cup_matches(df, year)
        if wc.empty:
            continue

        for i, (_, match) in enumerate(wc.iterrows()):
            if i % 10 == 0:
                print(f"    WC {year}: partido {i}/{len(wc)}", flush=True)

            match_date = str(match["date"])[:10]
            train = df[df["date"] < match["date"]].copy()
            if train.empty:
                continue

            elo_lookup = get_elo_at(timeline, match_date)
            hn = normalize_team_name(match["home_team"])
            an = normalize_team_name(match["away_team"])

            # Calcular strengths con xG real si está disponible
            strengths = compute_weighted_strengths(
                train,
                elo_lookup=elo_lookup,
                elo_sigma=params["elo_sigma"],
                recency_half_life_days=params["recency_half_life_days"],
                shrinkage_matches=params["shrinkage_matches"],
                min_weighted_matches=params["min_weighted_matches"],
                use_xg_real=use_xg,
            )

            hr = strengths[strengths["team"] == hn]
            ar = strengths[strengths["team"] == an]
            if hr.empty or ar.empty:
                continue

            h = TeamStrength(
                name=hn,
                attack=float(hr["attack"].iloc[0]),
                defense_vulnerability=float(hr["defense_vulnerability"].iloc[0]),
            )
            a = TeamStrength(
                name=an,
                attack=float(ar["attack"].iloc[0]),
                defense_vulnerability=float(ar["defense_vulnerability"].iloc[0]),
            )

            home_elo = elo_lookup.get(hn, 1500.0)
            away_elo = elo_lookup.get(an, 1500.0)

            p_h, p_d, p_a = model.predict(
                h, a, home_elo=home_elo, away_elo=away_elo, use_xg=use_xg
            )

            all_preds.append((p_h, p_d, p_a))

            home_goals = int(match["home_goals"])
            away_goals = int(match["away_goals"])
            if home_goals > away_goals:
                outcome = "H"
            elif home_goals < away_goals:
                outcome = "A"
            else:
                outcome = "D"
            all_outcomes.append(outcome)

            # Marcador más probable
            pred_score = (int(p_h * 3), int(p_a * 3))  # Aproximación simple
            all_pred_scores.append(pred_score)
            all_actual_scores.append((home_goals, away_goals))

    # Calcular métricas
    n = len(all_preds)
    if n == 0:
        return {"error": "No predictions"}

    brier = sum(brier_score(p, o) for p, o in zip(all_preds, all_outcomes)) / n
    rps = sum(ranked_probability_score(p, o) for p, o in zip(all_preds, all_outcomes)) / n
    ll = sum(log_loss(p, o) for p, o in zip(all_preds, all_outcomes)) / n

    sign_correct = sum(
        1 for p, o in zip(all_preds, all_outcomes)
        if np.argmax(p) == ["H", "D", "A"].index(o)
    )
    sign_acc = sign_correct / n

    return {
        "n": n,
        "brier": brier,
        "rps": rps,
        "log_loss": ll,
        "sign_accuracy": sign_acc,
    }


def main():
    print("=" * 80)
    print("ENSEMBLE BACKTEST - Progreso en tiempo real")
    print("=" * 80)
    print()

    # Cargar datos
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")

    print("Cargando timeline Elo...", flush=True)
    t0 = time.time()
    timeline = precompute_and_cache(csv_path, cache_path)
    print(f"  Timeline cargado en {time.time() - t0:.1f}s")

    print("Cargando dataset...", flush=True)
    t0 = time.time()
    df = load_martj42_csv(csv_path)
    print(f"  Dataset cargado en {time.time() - t0:.1f}s")

    # Parámetros base (del Sprint 2)
    settings = get_settings()
    params = {
        "elo_sigma": settings.elo_sigma,
        "recency_half_life_days": settings.recency_half_life_days,
        "shrinkage_matches": settings.shrinkage_matches,
        "min_weighted_matches": settings.min_weighted_matches,
        "draw_penalty_threshold": settings.draw_penalty_threshold,
        "draw_penalty_strength": settings.draw_penalty_strength,
        "elo_gap_inflation": settings.elo_gap_inflation,
        "draw_boost": settings.draw_boost,
    }

    print()
    print("Parámetros base:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()

    # Configurar pesos del ensemble
    configs = [
        # (weight_poisson, weight_bivariate, weight_xg, rho_bivariate, use_xg, label)
        (1.0, 0.0, 0.0, 0.0, False, "Poisson base"),
        (0.0, 1.0, 0.0, 0.05, False, "Bivariate Poisson (rho=0.05)"),
        (0.0, 1.0, 0.0, 0.10, False, "Bivariate Poisson (rho=0.10)"),
        (0.5, 0.5, 0.0, 0.05, False, "Ensemble 50/50 (Poisson+Biv, rho=0.05)"),
        (0.5, 0.5, 0.0, 0.10, False, "Ensemble 50/50 (Poisson+Biv, rho=0.10)"),
        (0.6, 0.4, 0.0, 0.05, False, "Ensemble 60/40 (Poisson+Biv, rho=0.05)"),
        (0.7, 0.3, 0.0, 0.05, False, "Ensemble 70/30 (Poisson+Biv, rho=0.05)"),
        (0.4, 0.4, 0.2, 0.05, True, "Ensemble 40/40/20 (con xG)"),
        (0.5, 0.3, 0.2, 0.05, True, "Ensemble 50/30/20 (con xG)"),
    ]

    results = []
    total_configs = len(configs)

    for idx, (w_p, w_b, w_xg, rho, use_xg, label) in enumerate(configs, 1):
        print(f"[{idx}/{total_configs}] {label}", flush=True)
        print(f"  Pesos: Poisson={w_p:.2f}, Bivariate={w_b:.2f}, xG={w_xg:.2f}", flush=True)

        t0 = time.time()
        metrics = evaluate_ensemble(
            df, timeline, params,
            weight_poisson=w_p,
            weight_bivariate=w_b,
            weight_xg=w_xg,
            rho_bivariate=rho,
            use_xg=use_xg,
        )
        elapsed = time.time() - t0

        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            print(f"  Resultados ({elapsed:.1f}s):")
            print(f"    Brier: {metrics['brier']:.4f}")
            print(f"    RPS: {metrics['rps']:.4f}")
            print(f"    Log loss: {metrics['log_loss']:.4f}")
            print(f"    Sign accuracy: {metrics['sign_accuracy']*100:.1f}%")
            print(f"    Partidos evaluados: {metrics['n']}")

            results.append({
                "label": label,
                "weight_poisson": w_p,
                "weight_bivariate": w_b,
                "weight_xg": w_xg,
                "rho_bivariate": rho,
                "use_xg": use_xg,
                **metrics,
            })

        print()

        # Guardar checkpoint cada 3 configuraciones
        if idx % 3 == 0:
            checkpoint_path = Path("data/processed/ensemble_checkpoint.json")
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Checkpoint guardado: {checkpoint_path}")
            print()

    # Guardar resultados finales
    output_path = Path("data/processed/ensemble_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("RESULTADOS FINALES")
    print("=" * 80)
    print()

    # Ordenar por Brier
    results_sorted = sorted(results, key=lambda x: x["brier"])

    print(f"{'Modelo':<50} {'Brier':>8} {'RPS':>8} {'Sign%':>8}")
    print("-" * 80)
    for r in results_sorted:
        print(f"{r['label']:<50} {r['brier']:>8.4f} {r['rps']:>8.4f} {r['sign_accuracy']*100:>7.1f}%")

    print()
    print(f"Resultados guardados en: {output_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
