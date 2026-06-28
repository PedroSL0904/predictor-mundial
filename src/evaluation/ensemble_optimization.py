"""Optimizacion de pesos del Ensemble via LOO 2014+2018+2022.

Pasos:
1. Para cada mundial (2014, 2018, 2022), correr 3 modelos (Poisson, BP, Skellam)
   en cada partido. Guardar (probs_poisson, probs_bp, probs_skellam, outcomes).
2. LOO: para cada test_year, ajustar pesos con train_years = otros 2.
3. Promediar los 3 sets de pesos → pesos finales.

Optimizacion: grid search sobre simplex {w1+w2+w3=1, w_i >= 0} con paso 0.05
(21 puntos por eje, ~200 combinaciones efectivas). Metrica objetivo: Brier score.

Resultado: ~3% de mejora esperada sobre Poisson puro segun literatura
(Maher 1982, Karlis-Ntzoufras 2003), aunque depende de la métrica.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import get_elo_at, precompute_and_cache
from src.data.historical import load_martj42_csv, normalize_team_name
from src.domain import outcome_from_score
from src.evaluation.backtest_cached import get_world_cup_matches
from src.features.recent_form import (
    blend_recent_with_historical,
    compute_recent_form,
)
from src.features.strengths_cache import StrengthsCache
from src.logging_config import get_logger
from src.models import BivariatePoissonModel, PoissonGoalModel, SkellamModel, TeamStrength

logger = get_logger(__name__)


@dataclass
class EnsembleWeights:
    """Pesos optimizados del ensemble. Suman 1, todos >= 0."""

    poisson: float
    bivariate_poisson: float
    skellam: float
    brier_train: float  # Brier score en el set de train (promedio LOO)

    def as_list(self) -> list[float]:
        return [self.poisson, self.bivariate_poisson, self.skellam]

    def __str__(self) -> str:
        return (
            f"[P={self.poisson:.2f}, BP={self.bivariate_poisson:.2f}, "
            f"S={self.skellam:.2f}] (brier_train={self.brier_train:.4f})"
        )


def _build_models() -> tuple[PoissonGoalModel, BivariatePoissonModel, SkellamModel]:
    """Build los 3 modelos con los settings del sistema."""
    s = get_settings()
    common = {
        "draw_boost": s.draw_boost,
        "draw_penalty_threshold": s.draw_penalty_threshold,
        "draw_penalty_strength": s.draw_penalty_strength,
        "elo_gap_inflation": s.elo_gap_inflation,
        "league_avg_multiplier": 1.0,  # Backtest no aplica multiplier (datos historicos)
    }
    return (
        PoissonGoalModel(**common),
        BivariatePoissonModel(lambda_3=0.10, **common),
        SkellamModel(**common),
    )


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score multiclass. probs shape (N, 3), outcomes shape (N,)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcomes] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def collect_per_model_predictions(
    df: pd.DataFrame,
    year: int,
    cache: StrengthsCache,
    timeline: dict[str, dict[str, float]],
    verbose: bool = True,
    enable_historical_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Para cada partido del mundial `year`, devuelve probs de los 3 modelos.

    Args:
        enable_historical_features: si True, aplica H2H + momentum + WC history
            sobre los strengths antes de predecir. Default False para
            backward compat con Sprint A4 (que optimizo sin features).

    Returns:
        (probs_poisson, probs_bp, probs_skellam, outcomes) cada uno (N, 3) y (N,).
    """
    s = get_settings()
    p, bp, sk = _build_models()

    wc = get_world_cup_matches(df, year)
    if wc.empty:
        empty = np.empty((0, 3))
        return empty, empty, empty, np.empty((0,), dtype=np.int64)

    wc_sorted = wc.sort_values("date").reset_index(drop=True)
    first_date = str(wc_sorted["date"].iloc[0])[:10]
    cache.set_elo_snapshot(first_date)

    from src.features.historical_features import compute_match_features

    probs_p: list[list[float]] = []
    probs_bp: list[list[float]] = []
    probs_sk: list[list[float]] = []
    outs: list[int] = []

    total = len(wc_sorted)
    for i, (_, match) in enumerate(wc_sorted.iterrows()):
        if verbose and i % 16 == 0:
            logger.info(f"    [{year} {i}/{total}]")
        match_date = str(match["date"])[:10]
        home_norm = normalize_team_name(match["home_team"])
        away_norm = normalize_team_name(match["away_team"])

        strengths = cache.get_strengths(
            match_date,
            shrinkage_matches=s.shrinkage_matches,
            min_weighted_matches=s.min_weighted_matches,
        )
        if s.recent_form_n_matches > 0 and s.recent_form_weight > 0:
            train_recent = df[df["date"] < match["date"]].copy()
            recent = compute_recent_form(
                train_recent,
                as_of=match_date,
                n_matches=s.recent_form_n_matches,
                min_matches=min(3, s.recent_form_n_matches),
            )
            strengths = blend_recent_with_historical(
                strengths, recent, weight_recent=s.recent_form_weight,
            )

        h = strengths[strengths["team"] == home_norm]
        a = strengths[strengths["team"] == away_norm]
        if h.empty or a.empty:
            continue

        # Aplicar features historicas (H2H, momentum, WC history) si esta habilitado
        if enable_historical_features:
            h_att_hist, h_def_hist, a_att_hist, a_def_hist = compute_match_features(
                df, home_norm, away_norm, match_date, enable=True,
            )
        else:
            h_att_hist = h_def_hist = a_att_hist = a_def_hist = 1.0

        home = TeamStrength(
            name=home_norm,
            attack=float(h["attack"].iloc[0]) * h_att_hist,
            defense_vulnerability=float(h["defense_vulnerability"].iloc[0]) * h_def_hist,
        )
        away = TeamStrength(
            name=away_norm,
            attack=float(a["attack"].iloc[0]) * a_att_hist,
            defense_vulnerability=float(a["defense_vulnerability"].iloc[0]) * a_def_hist,
        )

        elo_lookup = get_elo_at(timeline, match_date)
        home_elo = elo_lookup.get(home_norm, ORIGINAL_ELO)
        away_elo = elo_lookup.get(away_norm, ORIGINAL_ELO)

        pred_p = p.predict(home, away, home_elo=home_elo, away_elo=away_elo)
        pred_bp = bp.predict(home, away, home_elo=home_elo, away_elo=away_elo)
        pred_sk = sk.predict(home, away, home_elo=home_elo, away_elo=away_elo)

        probs_p.append([pred_p.p_home, pred_p.p_draw, pred_p.p_away])
        probs_bp.append([pred_bp.p_home, pred_bp.p_draw, pred_bp.p_away])
        probs_sk.append([pred_sk.p_home, pred_sk.p_draw, pred_sk.p_away])
        outs.append(
            outcome_from_score(int(match["home_goals"]), int(match["away_goals"]))
            .value
        )
        # Map "H"/"D"/"A" to 0/1/2
        outs[-1] = {"H": 0, "D": 1, "A": 2}[outs[-1]]

    if verbose:
        logger.info("done")

    return (
        np.array(probs_p),
        np.array(probs_bp),
        np.array(probs_sk),
        np.array(outs, dtype=np.int64),
    )


def optimize_weights_for_set(
    probs_p: np.ndarray,
    probs_bp: np.ndarray,
    probs_sk: np.ndarray,
    outcomes: np.ndarray,
    step: float = 0.05,
    metric: str = "brier",
) -> tuple[tuple[float, float, float], float]:
    """Grid search sobre simplex {w1+w2+w3=1, w_i >= 0}.

    Returns:
        ((w_poisson, w_bp, w_skellam), best_metric).
    """
    if len(outcomes) == 0:
        return (1.0, 0.0, 0.0), float("inf")

    best: tuple[tuple[float, float, float], float] = ((1.0, 0.0, 0.0), float("inf"))
    w1 = 0.0
    while w1 <= 1.0 + 1e-9:
        w2 = 0.0
        while w1 + w2 <= 1.0 + 1e-9:
            w3 = 1.0 - w1 - w2
            ens = w1 * probs_p + w2 * probs_bp + w3 * probs_sk
            if metric == "brier":
                m = _brier(ens, outcomes)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            if m < best[1]:
                best = ((round(w1, 4), round(w2, 4), round(w3, 4)), m)
            w2 += step
        w1 += step
    return best


def loo_optimize_ensemble(
    df: pd.DataFrame,
    cache: StrengthsCache,
    timeline: dict[str, dict[str, float]],
    years: tuple[int, ...] = (2014, 2018, 2022),
    step: float = 0.05,
    verbose: bool = True,
    enable_historical_features: bool = False,
) -> EnsembleWeights:
    """LOO 3 mundial: para cada año, entrena con los otros 2 y promedia.

    Args:
        enable_historical_features: si True, las probs de cada modelo se
            computan con H2H + momentum + WC history aplicadas. Sprint A4b.

    Returns:
        EnsembleWeights con los pesos finales y el brier promedio en train.
    """
    # Precompute predictions para todos los años
    if verbose:
        logger.info("Precomputando predicciones por modelo y mundial...")
    preds_per_year: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for y in years:
        t0 = time.time()
        if verbose:
            logger.info(f"  WC {y}:")
        probs_p, probs_bp, probs_sk, outs = collect_per_model_predictions(
            df, y, cache, timeline, verbose=False,
            enable_historical_features=enable_historical_features,
        )
        preds_per_year[y] = (probs_p, probs_bp, probs_sk, outs)
        if verbose:
            logger.info(f"{len(outs)} partidos en {time.time()-t0:.1f}s")

    # LOO: para cada test_year, fit en train_years
    loo_weights: list[tuple[float, float, float]] = []
    loo_briers: list[float] = []
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        probs_p_train = np.vstack([preds_per_year[y][0] for y in train_years])
        probs_bp_train = np.vstack([preds_per_year[y][1] for y in train_years])
        probs_sk_train = np.vstack([preds_per_year[y][2] for y in train_years])
        outcomes_train = np.concatenate([preds_per_year[y][3] for y in train_years])

        weights, brier = optimize_weights_for_set(
            probs_p_train,
            probs_bp_train,
            probs_sk_train,
            outcomes_train,
            step=step,
        )
        if verbose:
            logger.info(f"  LOO test={test_year}: weights={weights} brier={brier:.4f}")
        loo_weights.append(weights)
        loo_briers.append(brier)

    avg = tuple(np.mean(loo_weights, axis=0))
    avg_brier = float(np.mean(loo_briers))
    # Renormalizar para garantizar suma=1 (puede haber drift por promedio)
    total = sum(avg)
    avg = tuple(w / total for w in avg)

    return EnsembleWeights(
        poisson=avg[0],
        bivariate_poisson=avg[1],
        skellam=avg[2],
        brier_train=avg_brier,
    )


def evaluate_ensemble_on_year(
    df: pd.DataFrame,
    cache: StrengthsCache,
    timeline: dict[str, dict[str, float]],
    weights: EnsembleWeights,
    year: int,
) -> dict:
    """Evalua los pesos del ensemble en un mundial especifico."""
    probs_p, probs_bp, probs_sk, outs = collect_per_model_predictions(
        df, year, cache, timeline, verbose=False
    )
    if len(outs) == 0:
        return {"year": year, "n": 0}
    ens = (
        weights.poisson * probs_p
        + weights.bivariate_poisson * probs_bp
        + weights.skellam * probs_sk
    )
    brier = _brier(ens, outs)
    return {
        "year": year,
        "n": int(len(outs)),
        "brier_ensemble": brier,
        "brier_poisson_only": _brier(probs_p, outs),
        "brier_bp_only": _brier(probs_bp, outs),
        "brier_skellam_only": _brier(probs_sk, outs),
    }


def main() -> None:
    """CLI: corre LOO + evalua en cada mundial."""
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.parquet")
    logger.info("Cargando datos...")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)

    logger.info("\n=== LOO 3 mundial ===")
    t0 = time.time()
    weights = loo_optimize_ensemble(df, cache, timeline)
    logger.info(f"\nPesos finales (promedio LOO): {weights}")
    logger.info(f"Total: {time.time()-t0:.1f}s\n")

    logger.info("=== Evaluacion por mundial ===")
    for y in (2014, 2018, 2022):
        m = evaluate_ensemble_on_year(df, cache, timeline, weights, y)
        logger.info(f"  WC {y}: brier_ens={m.get('brier_ensemble', 0):.4f} "
            f"(P={m.get('brier_poisson_only', 0):.4f}, "
            f"BP={m.get('brier_bp_only', 0):.4f}, "
            f"S={m.get('brier_skellam_only', 0):.4f})")


if __name__ == "__main__":
    main()
