"""Modulo de calibracion de probabilidades.

Tecnicas implementadas:
- Temperature scaling: 1 parametro T que suaviza/agudiza probs.
  Entrena minimizando NLL con backtest historico.
  Persiste en disco para reutilizar en cada prediccion.

- Platt scaling: regresion logistica (multinomial). 2 params por outcome.
  Mayor riesgo de overfitting con pocos datos.

- Isotonic regression: muy flexible, requiere >1000 samples.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


class TemperatureScaler:
    """Calibrador con 1 parametro T.

    p_calibrated = softmax(logits / T)
    T > 1 -> suaviza (acerca al uniforme)
    T < 1 -> agudiza (acerca al top-1)
    T = 1 -> identidad
    """

    def __init__(self) -> None:
        self.T_: float = 1.0
        self.fitted: bool = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "TemperatureScaler":
        eps = 1e-9
        probs_c = np.clip(probs, eps, 1 - eps)
        probs_c = probs_c / probs_c.sum(axis=1, keepdims=True)
        logits = np.log(probs_c)

        def nll(T: float) -> float:
            if T < 0.01:
                T = 0.01
            p = _softmax(logits / T, axis=1)
            p = np.clip(p, eps, 1 - eps)
            return -np.log(p[np.arange(len(p)), outcomes]).mean()

        result = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
        self.T_ = float(result.x)
        self.fitted = True
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return probs
        eps = 1e-9
        probs_c = np.clip(probs, eps, 1 - eps)
        probs_c = probs_c / probs_c.sum(axis=1, keepdims=True)
        logits = np.log(probs_c)
        return _softmax(logits / self.T_, axis=1)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"T": self.T_, "fitted": self.fitted}, f)

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        obj.T_ = data["T"]
        obj.fitted = data["fitted"]
        return obj


class PlattCalibrator:
    """Platt scaling multinomial (sklearn)."""

    def __init__(self) -> None:
        self.model_ = None
        self.fitted: bool = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "PlattCalibrator":
        eps = 1e-9
        probs_clipped = np.clip(probs, eps, 1 - eps)
        probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
        logits = np.log(probs_clipped)
        self.model_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.model_.fit(logits, outcomes)
        self.fitted = True
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return probs
        eps = 1e-9
        probs_clipped = np.clip(probs, eps, 1 - eps)
        probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
        logits = np.log(probs_clipped)
        return self.model_.predict_proba(logits)


def collect_raw_probs(
    df: pd.DataFrame,
    year: int,
    cache = None,
    timeline = None,
    rf_n: int | None = None,
    rf_w: float | None = None,
    draw_boost: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Corre el backtest y devuelve (probs Nx3, outcomes N) para un mundial."""
    from src.config import get_settings
    from src.features.recent_form import blend_recent_with_historical, compute_recent_form
    from src.evaluation.backtest import get_world_cup_matches
    from src.features.strengths_cache import StrengthsCache
    from src.models import PoissonGoalModel, TeamStrength
    from src.evaluation.backtest_elo import get_elo_at
    from src.data.elo import ORIGINAL_ELO

    settings = get_settings()
    if rf_n is None:
        rf_n = settings.recent_form_n_matches
    if rf_w is None:
        rf_w = settings.recent_form_weight
    if draw_boost is None:
        draw_boost = settings.draw_boost

    if cache is None or timeline is None:
        from src.data.elo_timeline import precompute_and_cache
        from src.data.historical import load_martj42_csv
        csv_path = Path(r"C:\dev\predictor-mundial\data\raw\martj42_results.csv")
        cache_path = Path(r"C:\dev\predictor-mundial\data\processed\elo_timeline.json")
        timeline = precompute_and_cache(csv_path, cache_path)
        df_all = load_martj42_csv(csv_path)
        cache = StrengthsCache(df_all, timeline)

    wc = get_world_cup_matches(df, year)
    if wc.empty:
        return np.array([]), np.array([])

    first_date = str(wc["date"].min())[:10]
    cache.set_elo_snapshot(first_date)

    probs_list = []
    outcomes_list = []
    for _, match in wc.iterrows():
        match_date = str(match["date"])[:10]
        strengths = cache.get_strengths(
            match_date,
            shrinkage_matches=settings.shrinkage_matches,
            min_weighted_matches=settings.min_weighted_matches,
        )
        train = df[df["date"] < match["date"]].copy()
        if rf_n > 0 and rf_w > 0:
            recent = compute_recent_form(
                train, as_of=match_date, n_matches=rf_n,
                min_matches=min(3, rf_n),
            )
            strengths = blend_recent_with_historical(
                strengths, recent, weight_recent=rf_w,
            )
        h = strengths[strengths["team"] == match["home_team"]]
        a = strengths[strengths["team"] == match["away_team"]]
        if h.empty or a.empty:
            continue
        home = TeamStrength(
            name=match["home_team"],
            attack=float(h["attack"].iloc[0]),
            defense_vulnerability=float(h["defense_vulnerability"].iloc[0]),
        )
        away = TeamStrength(
            name=match["away_team"],
            attack=float(a["attack"].iloc[0]),
            defense_vulnerability=float(a["defense_vulnerability"].iloc[0]),
        )
        elo_lookup = get_elo_at(timeline, match_date)
        home_elo = elo_lookup.get(match["home_team"], ORIGINAL_ELO)
        away_elo = elo_lookup.get(match["away_team"], ORIGINAL_ELO)
        model = PoissonGoalModel(
            draw_penalty_threshold=settings.draw_penalty_threshold,
            draw_penalty_strength=settings.draw_penalty_strength,
            elo_gap_inflation=settings.elo_gap_inflation,
            draw_boost=draw_boost,
        )
        pred = model.predict(home, away, home_elo=home_elo, away_elo=away_elo)
        probs_list.append([pred.p_home, pred.p_draw, pred.p_away])
        hg, ag = int(match["home_goals"]), int(match["away_goals"])
        if pd.isna(hg) or pd.isna(ag):
            continue
        if hg > ag:
            outcomes_list.append(0)
        elif hg < ag:
            outcomes_list.append(2)
        else:
            outcomes_list.append(1)

    return np.array(probs_list), np.array(outcomes_list)


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcomes] = 1
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.log(np.maximum(probs[np.arange(len(probs)), outcomes], eps)).mean())


def sign_acc(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((np.argmax(probs, axis=1) == outcomes).mean())


def train_temperature_scaler() -> TemperatureScaler:
    """Entrena el TemperatureScaler con backtest historico (2014/2018/2022).

    LOO: para cada mundial, entrena con los otros 2 y promedia T.
    """
    from src.config import get_settings
    from src.data.elo_timeline import precompute_and_cache
    from src.data.historical import load_martj42_csv
    from src.features.strengths_cache import StrengthsCache

    settings = get_settings()
    csv_path = Path(r"C:\dev\predictor-mundial\data\raw\martj42_results.csv")
    cache_path = Path(r"C:\dev\predictor-mundial\data\processed\elo_timeline.json")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    years = [2014, 2018, 2022]
    Ts = []
    for test_year in years:
        train_years = [y for y in years if y != test_year]
        train_probs, train_out = [], []
        for ty in train_years:
            cache = StrengthsCache(df, timeline)
            probs, outcomes = collect_raw_probs(
                df, ty, cache, timeline,
                rf_n=settings.recent_form_n_matches,
                rf_w=settings.recent_form_weight,
                draw_boost=settings.draw_boost,
            )
            train_probs.append(probs)
            train_out.append(outcomes)
        train_probs = np.vstack(train_probs)
        train_out = np.concatenate(train_out)
        ts = TemperatureScaler()
        ts.fit(train_probs, train_out)
        Ts.append(ts.T_)

    # T promedio
    final_ts = TemperatureScaler()
    final_ts.T_ = float(np.mean(Ts))
    final_ts.fitted = True
    return final_ts
