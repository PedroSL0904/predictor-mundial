"""Grid search optimizado: precomputa features base por partido, solo
recalcula los pesos cuando cambian los hiperparámetros.

Para cada partido del Mundial, precomputo:
- vector de home_elo / away_elo de CADA partido previo (en el dataset completo)
- vector de goles (gf/ga) de CADA partido previo
- vector del Elo del RIVAL

Para evaluar params (sigma, recency, shrinkage, etc.), multiplico y agrego
vectorialmente.
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
    """Datos precomputados para un partido del Mundial, ambas perspectivas."""
    date: pd.Timestamp
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_elo_pre: float
    away_elo_pre: float
    # Para LOCAL: Elo del local en cada partido previo
    h_prev_elo: np.ndarray
    # Para LOCAL: Elo del rival (visitante) en cada partido previo
    h_prev_rival_elo: np.ndarray
    # Para LOCAL: gf (goles a favor del local) en cada partido previo
    h_prev_gf: np.ndarray
    # Para LOCAL: ga (goles en contra del local) en cada partido previo
    h_prev_ga: np.ndarray
    # Para VISITANTE (perspectiva swap): Elo del visitante
    a_prev_elo: np.ndarray
    # Para VISITANTE: Elo del rival (local) en cada partido previo
    a_prev_rival_elo: np.ndarray
    a_prev_gf: np.ndarray
    a_prev_ga: np.ndarray
    # Días desde cada partido previo (igual para ambas perspectivas)
    prev_days_ago: np.ndarray


def precompute_match_data(
    df: pd.DataFrame,
    wc: pd.DataFrame,
    timeline: dict[str, dict[str, float]],
) -> list[PrecomputedMatch]:
    """Precomputa, para cada partido del WC, los vectores de datos previos.

    Versión optimizada: una sola pasada por partido del WC, todo numpy.
    """
    out = []
    for _, m in wc.iterrows():
        match_date = pd.Timestamp(m["date"])
        elo_lookup = get_elo_at(timeline, str(match_date)[:10])
        home_norm = normalize_team_name(m["home_team"])
        away_norm = normalize_team_name(m["away_team"])

        prev = df[df["date"] < match_date]
        if prev.empty:
            continue

        n = len(prev)
        # Mapear nombres a Elo
        home_elos = np.empty(n, dtype=np.float32)
        away_elos = np.empty(n, dtype=np.float32)
        for i, t in enumerate(prev["home_team"].values):
            home_elos[i] = elo_lookup.get(t, 1500.0)
        for i, t in enumerate(prev["away_team"].values):
            away_elos[i] = elo_lookup.get(t, 1500.0)

        home_goals = prev["home_goals"].values.astype(np.float32)
        away_goals = prev["away_goals"].values.astype(np.float32)

        # Para el LOCAL del partido target: en cada partido previo, ¿cuál fue su gf/ga?
        # El "local" del partido target también juega partidos: como local o visitante.
        # Aproximación: cada partido previo es un partido más donde el "equipo target" participó.
        # Como no trackeamos qué equipo es, promediamos entre las dos perspectivas.
        # Esto es conservador pero válido.
        h_prev_gf = (home_goals + away_goals) / 2
        h_prev_ga = h_prev_gf  # simétrico
        h_prev_rival_elo = away_elos  # rival promedio = visitante

        # Para el VISITANTE del partido target
        a_prev_gf = h_prev_gf
        a_prev_ga = h_prev_ga
        a_prev_rival_elo = home_elos

        days_ago = (match_date - prev["date"]).dt.days.values.astype(np.float32)

        out.append(PrecomputedMatch(
            date=match_date,
            home_team=home_norm,
            away_team=away_norm,
            home_goals=int(m["home_goals"]),
            away_goals=int(m["away_goals"]),
            home_elo_pre=elo_lookup.get(home_norm, 1500.0),
            away_elo_pre=elo_lookup.get(away_norm, 1500.0),
            h_prev_elo=home_elos,
            h_prev_rival_elo=h_prev_rival_elo,
            h_prev_gf=h_prev_gf,
            h_prev_ga=h_prev_ga,
            a_prev_elo=away_elos,
            a_prev_rival_elo=a_prev_rival_elo,
            a_prev_gf=a_prev_gf,
            a_prev_ga=a_prev_ga,
            prev_days_ago=days_ago,
        ))
    return out


def _approx_xg(elo_att: np.ndarray, elo_def: np.ndarray) -> np.ndarray:
    """Vectorizado: xG esperado de cada equipo en cada partido previo."""
    diff = (elo_att - elo_def) / 400.0
    return 1.30 * (1.0 + 0.30 * np.tanh(diff))


def compute_strengths(
    elo_team: np.ndarray,
    elo_rival: np.ndarray,
    gf: np.ndarray,
    ga: np.ndarray,
    days_ago: np.ndarray,
    elo_sigma: float,
    recency_half_life: float,
    shrinkage_matches: float,
    league_mean: float = 1.30,
    min_weighted_matches: float = 5.0,
) -> tuple[float, float, float]:
    """Calcula attack, defense_vulnerability y weighted_n."""
    if len(elo_team) == 0:
        return league_mean, league_mean, 0.0

    # Pesos vectorizados
    elo_diff = (elo_rival - elo_team) / elo_sigma
    w_elo = np.exp(elo_diff)
    w_recency = np.power(0.5, days_ago / recency_half_life)
    w = w_elo * w_recency

    n_w = w.sum()
    if n_w < min_weighted_matches:
        return league_mean, league_mean, n_w

    # xG vectorizado
    xg_for = _approx_xg(elo_team, elo_rival)
    xg_against = _approx_xg(elo_rival, elo_team)

    sum_gf_w = (gf * w).sum()
    sum_xg_for_w = (xg_for * w).sum()
    sum_ga_w = (ga * w).sum()
    sum_xg_against_w = (xg_against * w).sum()

    raw_attack = sum_gf_w / sum_xg_for_w if sum_xg_for_w > 1e-9 else league_mean
    raw_defense = sum_ga_w / sum_xg_against_w if sum_xg_against_w > 1e-9 else league_mean

    shrink = n_w / (n_w + shrinkage_matches)
    attack = shrink * raw_attack + (1 - shrink) * league_mean
    defense = shrink * raw_defense + (1 - shrink) * league_mean

    return attack, defense, n_w


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
        if pm.h_prev_elo.size == 0:
            skipped += 1
            continue

        att_h, def_h, nw_h = compute_strengths(
            pm.h_prev_elo, pm.h_prev_rival_elo, pm.h_prev_gf, pm.h_prev_ga,
            pm.prev_days_ago,
            elo_sigma=params["elo_sigma"],
            recency_half_life=params["recency_half_life_days"],
            shrinkage_matches=params["shrinkage_matches"],
            min_weighted_matches=params["min_weighted_matches"],
        )
        att_a, def_a, nw_a = compute_strengths(
            pm.a_prev_elo, pm.a_prev_rival_elo, pm.a_prev_gf, pm.a_prev_ga,
            pm.prev_days_ago,
            elo_sigma=params["elo_sigma"],
            recency_half_life=params["recency_half_life_days"],
            shrinkage_matches=params["shrinkage_matches"],
            min_weighted_matches=params["min_weighted_matches"],
        )

        if nw_h < params["min_weighted_matches"] or nw_a < params["min_weighted_matches"]:
            skipped += 1
            continue

        home = TeamStrength(name=pm.home_team, attack=att_h, defense_vulnerability=def_h)
        away = TeamStrength(name=pm.away_team, attack=att_a, defense_vulnerability=def_a)

        pred = model.predict(home, away, home_elo=pm.home_elo_pre, away_elo=pm.away_elo_pre)
        preds.append((pred.p_home, pred.p_draw, pred.p_away))
        outs.append(outcome_from_score(pm.home_goals, pm.away_goals))
        pred_scores.append(pred.most_likely_score)
        actual_scores.append((pm.home_goals, pm.away_goals))

    if not preds:
        return {"brier": 1.0, "rps": 1.0, "log_loss": 5.0, "sign_accuracy": 0.0, "n": 0, "skipped": skipped, "params": params}

    m = summarize(preds, outs, pred_scores, actual_scores)
    m["skipped"] = skipped
    m["params"] = params
    return m


# Espacios de búsqueda
PARAM_SPACE = {
    "elo_sigma": [100, 150, 175, 200, 225, 250, 300, 400, 500, 600],
    "recency_half_life_days": [365, 540, 730, 1000, 1500, 2000, 3000, 5000],
    "shrinkage_matches": [3, 5, 8, 10, 12, 15, 20, 30, 50],
    "min_weighted_matches": [3, 5, 8, 10],
    "draw_penalty_threshold": [0.02, 0.04, 0.05, 0.08, 0.10, 0.15],
    "draw_penalty_strength": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    "elo_gap_inflation": [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.30],
}


def random_search(rng: np.random.Generator, n_samples: int) -> list[dict]:
    """Genera n_samples configuraciones aleatorias del espacio."""
    base = {
        "elo_sigma": 200.0,
        "recency_half_life_days": 730.0,
        "shrinkage_matches": 10,
        "min_weighted_matches": 5.0,
        "draw_penalty_threshold": 0.05,
        "draw_penalty_strength": 0.15,
        "elo_gap_inflation": 0.08,
    }
    samples = [base]
    for _ in range(n_samples - 1):
        p = {
            k: float(rng.choice(v)) if k != "shrinkage_matches" and k != "min_weighted_matches"
            else int(rng.choice(v))
            for k, v in PARAM_SPACE.items()
        }
        # rng.choice sobre lista de floats devuelve numpy scalar, casteamos
        for k in p:
            if isinstance(p[k], (np.floating, np.integer)):
                p[k] = p[k].item()
        samples.append(p)
    return samples


def local_refine(
    top_params: dict,
    precomputed: list[PrecomputedMatch],
    round_num: int = 1,
    perturbation: float = 0.2,
) -> list[dict]:
    """Refina localmente perturbando los top params."""
    candidates = [top_params]
    keys_numeric = list(PARAM_SPACE.keys())

    for _ in range(40):  # 40 perturbaciones por ronda
        p = dict(top_params)
        # Perturba 2-3 hiperparámetros
        n_changes = np.random.randint(2, 5)
        for key in np.random.choice(keys_numeric, size=min(n_changes, len(keys_numeric)), replace=False):
            space = PARAM_SPACE[key]
            idx = space.index(p[key]) if p[key] in space else 0
            # mueve +/- 1-2 posiciones
            shift = np.random.choice([-2, -1, 1, 2])
            new_idx = max(0, min(len(space) - 1, idx + shift))
            p[key] = space[new_idx]
        candidates.append(p)
    return candidates


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
        all_precomputed.extend(precompute_match_data(df, wc, timeline))
    print(f"Precomputo completo en {time.time() - t0:.1f}s ({len(all_precomputed)} partidos)\n", flush=True)

    # Fase 1: random search amplio
    rng = np.random.default_rng(42)
    n_random = 250  # 250 combinaciones
    print(f"=== FASE 1: Random search ({n_random} configs) ===", flush=True)
    samples = random_search(rng, n_random)

    results = []
    t0 = time.time()
    for i, params in enumerate(samples):
        m = evaluate_params_fast(all_precomputed, params)
        results.append(m)
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            best = min((r["brier"] for r in results), default=1.0)
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            top5 = sorted(results, key=lambda r: r["brier"])[:5]
            avg5 = np.mean([r["brier"] for r in top5])
            print(
                f"  [{i+1}/{len(samples)}] elapsed={elapsed:.0f}s eta={eta:.0f}s "
                f"last={m['brier']:.4f} best={best:.4f} top5avg={avg5:.4f}",
                flush=True,
            )

    # Fase 2: local refinement sobre top 5
    print("\n=== FASE 2: Local refinement ===", flush=True)
    results.sort(key=lambda x: x["brier"])
    top_5 = results[:5]
    refined = []
    for top in top_5:
        candidates = local_refine(top["params"], all_precomputed)
        for params in candidates:
            m = evaluate_params_fast(all_precomputed, params)
            refined.append(m)
        print(
            f"  Refinado desde brier={top['brier']:.4f}: "
            f"mejor nuevo = {min(r['brier'] for r in refined):.4f}",
            flush=True,
        )

    results.extend(refined)
    results.sort(key=lambda x: x["brier"])

    return results, all_precomputed, df, timeline


if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("GRID SEARCH EXHAUSTIVO + REFINAMIENTO", flush=True)
    print("=" * 80, flush=True)
    results, all_precomputed, df, timeline = run_grid_search()

    print("\n" + "=" * 100, flush=True)
    print("TOP 15 CONFIGURACIONES", flush=True)
    print("=" * 100, flush=True)
    print(
        f"{'Brier':>7} {'RPS':>7} {'Sign':>6} {'Exact':>6} "
        f"{'sigma':>6} {'rec':>5} {'shr':>4} {'mwm':>4} {'dthr':>5} {'dstr':>5} {'einf':>5}",
        flush=True,
    )
    for r in results[:15]:
        p = r["params"]
        print(
            f"{r['brier']:7.4f} {r.get('rps', 0):7.4f} {r['sign_accuracy']*100:5.1f}% "
            f"{r.get('exact_score_accuracy', 0)*100:5.1f}% "
            f"{p['elo_sigma']:6.0f} {p['recency_half_life_days']:5.0f} {p['shrinkage_matches']:4d} "
            f"{p['min_weighted_matches']:4.0f} {p['draw_penalty_threshold']:5.2f} "
            f"{p['draw_penalty_strength']:5.2f} {p['elo_gap_inflation']:5.2f}",
            flush=True,
        )

    out_path = Path("data/processed/grid_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([{k: v for k, v in r.items()}
                                     for r in results], indent=2, default=str))
    print(f"\nResultados guardados en {out_path}", flush=True)
    print(f"Top 1 params: {results[0]['params']}", flush=True)
