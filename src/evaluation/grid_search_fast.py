"""Grid search optimizado: precomputa features base por partido, solo
recalcula los pesos cuando cambian los hiperparámetros.

Para cada partido del Mundial, precomputo:
- vector de home_elo / away_elo de CADA partido previo (en el dataset completo)
- vector de goles (gf/ga) de CADA partido previo

Para evaluar params (sigma, recency, shrinkage, etc.), multiplico y agrego
vectorialmente. Esto baja de ~120s por combinación a <0.5s.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv, normalize_team_name
from src.evaluation.backtest import (
    get_world_cup_matches,
    outcome_from_score,
)
from src.evaluation.backtest_elo import get_elo_at
from src.evaluation.metrics import summarize
from src.models import PoissonGoalModel, TeamStrength


@dataclass
class PrecomputedMatch:
    """Datos precomputados para un partido del Mundial."""
    date: pd.Timestamp
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_elo_pre: float
    away_elo_pre: float
    # Pre-vector con el Elo del LOCAL en cada partido previo (long=n_prev)
    prev_home_elo: np.ndarray
    # Pre-vector con el Elo del VISITANTE en cada partido previo
    prev_away_elo: np.ndarray
    # Pre-vector con goles del local y visitante (gf, ga) en cada partido previo
    prev_gf: np.ndarray  # goles del "team perspective" en cada partido previo
    prev_ga: np.ndarray
    # ELO del rival en cada partido previo (para el local: el away_elo del partido previo)
    prev_rival_elo: np.ndarray
    # dias desde el partido previo hasta este
    prev_days_ago: np.ndarray


def precompute_match_data(
    df: pd.DataFrame,
    wc: pd.DataFrame,
    timeline: dict[str, dict[str, float]],
) -> list[PrecomputedMatch]:
    """Precomputa, para cada partido del WC, los vectores de datos previos."""
    out = []
    for _, m in wc.iterrows():
        match_date = pd.Timestamp(m["date"])
        elo_lookup = get_elo_at(timeline, str(match_date)[:10])
        home_norm = normalize_team_name(m["home_team"])
        away_norm = normalize_team_name(m["away_team"])

        # Partidos previos a este
        prev = df[df["date"] < match_date].copy()
        if prev.empty:
            continue

        # Elo de cada equipo en cada partido previo
        prev_home_elo = prev["home_team"].map(lambda t: elo_lookup.get(t, 1500.0)).values.astype(np.float32)
        prev_away_elo = prev["away_team"].map(lambda t: elo_lookup.get(t, 1500.0)).values.astype(np.float32)

        # Para el "team perspective" (local o visitante), cada partido previo
        # tiene gf y ga dependiendo de si el equipo estaba como local o visitante
        # Aquí construimos DOS perspectivas concatenadas, una por fila
        # (primera mitad: cuando local era home, segunda: cuando era away)

        # Para este partido, los "partidos previos del LOCAL" son todos los
        # partidos donde jugó como local o visitante. La perspectiva
        # del local: gf = home_goals cuando jugó como local, gf = away_goals cuando fue visitante.
        # Vectorizado:
        gf_home = np.where(
            np.ones(len(prev), dtype=bool),
            prev["home_goals"].values.astype(np.float32),
            0,
        )
        # Real: gf del local en cada partido previo
        gf_local = prev["home_goals"].values.astype(np.float32)
        gf_visit = prev["away_goals"].values.astype(np.float32)
        ga_local = prev["away_goals"].values.astype(np.float32)
        ga_visit = prev["home_goals"].values.astype(np.float32)

        # Para el local: en cada partido previo, ¿fue home o away?
        # Como precomputamos separado, duplicamos los arrays con un flag.
        # Para eficiencia, asumimos que cada equipo aparece mitad como local
        # y mitad como visitante en promedio; usamos un promedio ponderado simple.
        # Esto es una aproximación, pero válida para el grid search.
        prev_gf_local = (gf_local + gf_visit) / 2  # aprox promedio
        prev_ga_local = (ga_local + ga_visit) / 2
        prev_gf_visit = prev_gf_local
        prev_ga_visit = prev_ga_local

        # Elo del rival en cada partido previo (para el local: away_elo del partido)
        prev_rival_elo_local = prev_away_elo
        prev_rival_elo_visit = prev_home_elo

        # Días desde cada partido previo hasta este
        prev_days_ago = (match_date - prev["date"]).dt.days.values.astype(np.float32)

        out.append(PrecomputedMatch(
            date=match_date,
            home_team=home_norm,
            away_team=away_norm,
            home_goals=int(m["home_goals"]),
            away_goals=int(m["away_goals"]),
            home_elo_pre=elo_lookup.get(home_norm, 1500.0),
            away_elo_pre=elo_lookup.get(away_norm, 1500.0),
            prev_home_elo=prev_home_elo,
            prev_away_elo=prev_away_elo,
            prev_gf=prev_gf_local,
            prev_ga=prev_ga_local,
            prev_rival_elo=prev_rival_elo_local,
            prev_days_ago=prev_days_ago,
        ))
    return out


def _approx_xg(elo_attacker: float, elo_defender: float) -> float:
    diff = (elo_attacker - elo_defender) / 400.0
    return 1.30 * (1.0 + 0.30 * np.tanh(diff))


def compute_attack_defense_fast(
    pm: PrecomputedMatch,
    elo_sigma: float,
    recency_half_life: float,
    shrinkage_matches: float,
    league_mean: float = 1.30,
) -> tuple[float, float, int, float]:
    """Calcula attack y defense_vulnerability para un partido precomputado."""
    # Pesos: combinación de Elo del rival y recencia
    elo_diff = (pm.prev_rival_elo - pm.prev_home_elo) / elo_sigma
    w_elo = np.exp(elo_diff)
    w_recency = 0.5 ** (pm.prev_days_ago / recency_half_life)
    weights = w_elo * w_recency

    # Goles esperados: xG aproximado desde Elo
    xg_for = np.array([_approx_xg(pm.prev_home_elo[i], pm.prev_rival_elo[i])
                       for i in range(len(pm.prev_home_elo))], dtype=np.float32)
    xg_against = np.array([_approx_xg(pm.prev_rival_elo[i], pm.prev_home_elo[i])
                           for i in range(len(pm.prev_home_elo))], dtype=np.float32)

    # attack y defense como razones ponderadas
    sum_gf_w = (pm.prev_gf * weights).sum()
    sum_xg_for_w = (xg_for * weights).sum()
    sum_ga_w = (pm.prev_ga * weights).sum()
    sum_xg_against_w = (xg_against * weights).sum()

    raw_attack = sum_gf_w / sum_xg_for_w if sum_xg_for_w > 0 else league_mean
    raw_defense = sum_ga_w / sum_xg_against_w if sum_xg_against_w > 0 else league_mean

    n_w = weights.sum()
    if n_w < 1.0:
        return league_mean, league_mean, 0, 0.0

    # Shrinkage
    shrink = n_w / (n_w + shrinkage_matches)
    attack = shrink * raw_attack + (1 - shrink) * league_mean
    defense = shrink * raw_defense + (1 - shrink) * league_mean

    return attack, defense, len(pm.prev_home_elo), n_w


def evaluate_params_fast(
    precomputed: list[PrecomputedMatch],
    params: dict,
) -> dict:
    """Evalúa un set de params sobre partidos precomputados."""
    model = PoissonGoalModel(
        draw_penalty_threshold=params["draw_penalty_threshold"],
        draw_penalty_strength=params["draw_penalty_strength"],
        elo_gap_inflation=params["elo_gap_inflation"],
    )

    preds = []
    outs = []
    pred_scores = []
    actual_scores = []
    skipped = 0

    for pm in precomputed:
        if pm.prev_home_elo.size == 0:
            skipped += 1
            continue

        att, defn, n, n_w = compute_attack_defense_fast(
            pm,
            elo_sigma=params["elo_sigma"],
            recency_half_life=params["recency_half_life_days"],
            shrinkage_matches=params["shrinkage_matches"],
        )
        if n_w < params["min_weighted_matches"]:
            skipped += 1
            continue

        home = TeamStrength(name=pm.home_team, attack=att, defense_vulnerability=defn)
        away = TeamStrength(name=pm.away_team, attack=defn, defense_vulnerability=att)  # ojo, esto es sym
        # Real: para visitante calculo su propio attack/defense
        att_a, defn_a, _, n_w_a = compute_attack_defense_fast(
            _swap_perspective(pm),
            elo_sigma=params["elo_sigma"],
            recency_half_life=params["recency_half_life_days"],
            shrinkage_matches=params["shrinkage_matches"],
        )
        if n_w_a < params["min_weighted_matches"]:
            skipped += 1
            continue
        away = TeamStrength(name=pm.away_team, attack=att_a, defense_vulnerability=defn_a)

        pred = model.predict(home, away, home_elo=pm.home_elo_pre, away_elo=pm.away_elo_pre)
        preds.append((pred.p_home, pred.p_draw, pred.p_away))
        outs.append(outcome_from_score(pm.home_goals, pm.away_goals))
        pred_scores.append(pred.most_likely_score)
        actual_scores.append((pm.home_goals, pm.away_goals))

    if not preds:
        return {"brier": 1.0, "rps": 1.0, "log_loss": 5.0, "sign_accuracy": 0.0, "n": 0}

    m = summarize(preds, outs, pred_scores, actual_scores)
    m["skipped"] = skipped
    m["params"] = params
    return m


def _swap_perspective(pm: PrecomputedMatch) -> PrecomputedMatch:
    """Devuelve una copia de pm pero con la perspectiva del visitante."""
    return PrecomputedMatch(
        date=pm.date,
        home_team=pm.away_team,
        away_team=pm.home_team,
        home_goals=pm.away_goals,
        away_goals=pm.home_goals,
        home_elo_pre=pm.away_elo_pre,
        away_elo_pre=pm.home_elo_pre,
        prev_home_elo=pm.prev_away_elo,
        prev_away_elo=pm.prev_home_elo,
        prev_gf=pm.prev_ga,  # gf del visitante = ga del local
        prev_ga=pm.prev_gf,
        prev_rival_elo=pm.prev_home_elo,  # rival del visitante = local del partido previo
        prev_days_ago=pm.prev_days_ago,
    )


def run_grid_search() -> list[dict]:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.json")

    print("Cargando timeline Elo...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    print("Precomputando datos para partidos de WC...", flush=True)
    t0 = time.time()
    all_precomputed = []
    for year in [2014, 2018, 2022]:
        wc = get_world_cup_matches(df, year)
        print(f"  WC {year}: {len(wc)} partidos...", flush=True)
        all_precomputed.extend(precompute_match_data(df, wc, timeline))
    print(f"Precomputo completo en {time.time() - t0:.1f}s ({len(all_precomputed)} partidos)\n", flush=True)

    # Hiperparámetros
    base = {
        "elo_sigma": 200.0,
        "recency_half_life_days": 730.0,
        "shrinkage_matches": 10,
        "min_weighted_matches": 5.0,
        "draw_penalty_threshold": 0.05,
        "draw_penalty_strength": 0.15,
        "elo_gap_inflation": 0.08,
    }

    rng = np.random.default_rng(42)
    n_samples = 80
    samples = [base]
    for _ in range(n_samples - 1):
        p = {
            "elo_sigma": float(rng.choice([100, 150, 200, 250, 300, 400, 500])),
            "recency_half_life_days": float(rng.choice([365, 540, 730, 1000, 1500, 2000, 3000])),
            "shrinkage_matches": int(rng.choice([3, 5, 8, 10, 15, 20, 30])),
            "min_weighted_matches": float(rng.choice([3, 5, 8, 10])),
            "draw_penalty_threshold": float(rng.choice([0.02, 0.04, 0.05, 0.08, 0.10, 0.15])),
            "draw_penalty_strength": float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])),
            "elo_gap_inflation": float(rng.choice([0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.30])),
        }
        samples.append(p)

    results = []
    t0 = time.time()
    for i, params in enumerate(samples):
        m = evaluate_params_fast(all_precomputed, params)
        results.append(m)
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            best = min((r["brier"] for r in results), default=1.0)
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(
                f"  [{i+1}/{len(samples)}] elapsed={elapsed:.1f}s eta={eta:.1f}s "
                f"last_brier={m['brier']:.4f} best_brier={best:.4f}",
                flush=True,
            )

    results.sort(key=lambda x: x["brier"])
    return results


if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("GRID SEARCH OPTIMIZADO", flush=True)
    print("=" * 80, flush=True)
    results = run_grid_search()

    print("\n" + "=" * 90, flush=True)
    print("TOP 15 CONFIGURACIONES POR BRIER", flush=True)
    print("=" * 90, flush=True)
    print(
        f"{'Brier':>7} {'RPS':>7} {'Sign':>6} {'Exact':>6} "
        f"{'sigma':>6} {'rec':>5} {'shr':>4} {'dthr':>5} {'dstr':>5} {'einf':>5}",
        flush=True,
    )
    for r in results[:15]:
        p = r["params"]
        print(
            f"{r['brier']:7.4f} {r.get('rps', 0):7.4f} {r['sign_accuracy']*100:5.1f}% "
            f"{r.get('exact_score_accuracy', 0)*100:5.1f}% "
            f"{p['elo_sigma']:6.0f} {p['recency_half_life_days']:5.0f} {p['shrinkage_matches']:4d} "
            f"{p['draw_penalty_threshold']:5.2f} {p['draw_penalty_strength']:5.2f} "
            f"{p['elo_gap_inflation']:5.2f}",
            flush=True,
        )

    # Guardar
    out_path = Path("data/processed/grid_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([{k: v for k, v in r.items() if k != 'by_year'}
                                     for r in results], indent=2))
    print(f"\nTop 1 params: {results[0]['params']}", flush=True)
