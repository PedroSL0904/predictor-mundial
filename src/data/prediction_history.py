"""Tracking persistente de predicciones en data/processed/predictions_history.csv.

Cada predicción hecha por el sistema se registra con:
- timestamp (cuando se hizo la predicción)
- match_date, home, away
- p_home, p_draw, p_away
- predicted_score
- outcome (H/D/A o None si no se jugo)
- home_score, away_score (si esta disponible)

Permite:
1. Auditoria: ver que predijo el sistema en cada fecha
2. Calibracion: comparar probs vs outcomes historicos
3. Drift: detectar degradacion del modelo

Uso:
    history = PredictionHistory()
    history.record(pred, match_date="2026-06-15", home="Argentina", away="France")
    history.update_outcome("2026-06-15", "Argentina", "France", 2, 1)
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain import MatchPrediction, outcome_from_score
from src.paths import PREDICTIONS_HISTORY


class PredictionHistory:
    """Tracker persistente de predicciones."""

    COLUMNS = [
        "recorded_at", "match_date", "home", "away",
        "p_home", "p_draw", "p_away",
        "predicted_score", "model",
        "home_score", "away_score", "outcome",
    ]

    def __init__(self, path: Path = PREDICTIONS_HISTORY) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            try:
                df = pd.read_csv(self.path)
                # Si el CSV no tiene las columnas esperadas, tratar como vacio
                expected_first_cols = {"recorded_at", "match_date", "home", "away"}
                if not expected_first_cols.issubset(set(df.columns)):
                    return pd.DataFrame(columns=self.COLUMNS)
                # Ensure all expected columns exist
                for col in self.COLUMNS:
                    if col not in df.columns:
                        df[col] = None
                # Force dtypes numericos
                for col in ("p_home", "p_draw", "p_away", "home_score", "away_score"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            except (pd.errors.EmptyDataError, pd.errors.ParserError, Exception):
                pass
        return pd.DataFrame(columns=self.COLUMNS)

    def _save(self) -> None:
        self._df.to_csv(self.path, index=False)

    def record(
        self,
        pred: MatchPrediction,
        match_date: str,
        home: str,
        away: str,
        timestamp: str | None = None,
        model: str | None = None,
    ) -> None:
        """Registra una prediccion. Si ya existe (mismo match_date+home+away), la pisa.

        Args:
            pred: MatchPrediction del modelo.
            match_date: fecha del partido (ISO).
            home, away: nombres de equipos.
            timestamp: cuando se hizo la prediccion. Si None, ahora.
            model: nombre del modelo (opcional, default: pred.model).
        """
        ts = timestamp or datetime.now(UTC).isoformat()
        m = model or pred.model
        new_row = {
            "recorded_at": ts,
            "match_date": match_date,
            "home": home,
            "away": away,
            "p_home": float(pred.p_home),
            "p_draw": float(pred.p_draw),
            "p_away": float(pred.p_away),
            "predicted_score": f"{pred.most_likely_score[0]}-{pred.most_likely_score[1]}",
            "model": m,
            "home_score": None,
            "away_score": None,
            "outcome": None,
        }
        # Si ya existe, pisa
        mask = (
            (self._df["match_date"] == match_date)
            & (self._df["home"] == home)
            & (self._df["away"] == away)
            & (self._df["model"] == m)
        )
        if mask.any():
            # Update: preservar home_score/away_score/outcome si ya existian
            existing = self._df.loc[mask].iloc[0]
            for k in ("home_score", "away_score", "outcome"):
                val = existing.get(k)
                if pd.notna(val):
                    new_row[k] = val
            self._df = self._df.loc[~mask]
        # Concatenar (con tipos consistentes)
        new_df = pd.DataFrame(
            [{
                "recorded_at": new_row["recorded_at"],
                "match_date": new_row["match_date"],
                "home": new_row["home"],
                "away": new_row["away"],
                "p_home": np.float64(new_row["p_home"]),
                "p_draw": np.float64(new_row["p_draw"]),
                "p_away": np.float64(new_row["p_away"]),
                "predicted_score": new_row["predicted_score"],
                "model": new_row["model"],
                "home_score": np.float64("nan") if new_row["home_score"] is None else float(new_row["home_score"]),
                "away_score": np.float64("nan") if new_row["away_score"] is None else float(new_row["away_score"]),
                "outcome": new_row["outcome"] if new_row["outcome"] is not None else pd.NA,
            }],
            columns=self.COLUMNS,
        )
        self._df = pd.concat([self._df, new_df], ignore_index=True)
        self._save()

    def update_outcome(
        self,
        match_date: str,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        model: str | None = None,
    ) -> int:
        """Actualiza el resultado de un partido. Retorna # de rows actualizadas.

        Si `model` es None, actualiza TODOS los modelos para ese match.
        """
        mask = (
            (self._df["match_date"] == match_date)
            & (self._df["home"] == home)
            & (self._df["away"] == away)
        )
        if model is not None:
            mask = mask & (self._df["model"] == model)
        n = int(mask.sum())
        if n == 0:
            return 0
        outcome = outcome_from_score(home_score, away_score).value
        self._df.loc[mask, "home_score"] = home_score
        self._df.loc[mask, "away_score"] = away_score
        self._df.loc[mask, "outcome"] = outcome
        self._save()
        return n

    def get_history(self) -> pd.DataFrame:
        """Retorna todo el historial."""
        return self._df.copy()

    def get_unmatched(self) -> pd.DataFrame:
        """Retorna predicciones sin outcome (partidos pendientes)."""
        return self._df[self._df["outcome"].isna()].copy()

    def get_with_outcome(self) -> pd.DataFrame:
        """Retorna predicciones con outcome (partidos finalizados)."""
        return self._df[self._df["outcome"].notna()].copy()

    def get_metrics(self) -> dict:
        """Calcula metricas sobre predicciones con outcome.

        Returns:
            dict con brier, log_loss, sign_acc, n, o vacio si no hay datos.
        """
        df = self.get_with_outcome()
        if df.empty:
            return {}
        eps = 1e-9
        # Asegurar float64
        probs = np.asarray(
            df[["p_home", "p_draw", "p_away"]].values, dtype=np.float64
        )
        outcome_map = {"H": 0, "D": 1, "A": 2}
        outs = np.asarray(
            df["outcome"].map(outcome_map).values, dtype=np.int64
        )

        onehot = np.zeros_like(probs)
        onehot[np.arange(len(probs)), outs] = 1
        brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
        # Evitar log(0) usando clip antes de log
        clipped = np.clip(probs[np.arange(len(probs)), outs], eps, 1.0)
        logloss = float(-np.log(clipped).mean())
        sign_acc = float((np.argmax(probs, axis=1) == outs).mean())
        return {"brier": brier, "log_loss": logloss, "sign_acc": sign_acc, "n": int(len(df))}

    def clear(self) -> None:
        """Borra el historial (peligroso, usar solo en tests)."""
        self._df = pd.DataFrame(columns=self.COLUMNS)
        self._save()
