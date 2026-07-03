"""StrengthsCache: cache incremental de strengths para backtest rapido.

Replica exactamente el comportamiento de compute_weighted_strengths
cuando se llama con un train = df[df["date"] < as_of].

Uso en backtest:
  cache = StrengthsCache.from_dataframe(df, timeline)
  cache.set_elo_snapshot(as_of_date)
  strengths = cache.get_strengths(as_of_date)

El cache precomputa los sums incrementalmente. Como el xG depende del
elo_lookup (snapshot en as_of), el cache se reconstruye en set_elo_snapshot.
Esto toma ~14s para 49k partidos, pero solo se hace una vez por Mundial.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import get_elo_at


def _approx_xg(elo_a: np.ndarray, elo_d: np.ndarray) -> np.ndarray:
    """DEPRECATED: usa src.features.xg_approximation.approx_xg."""
    from src.features.xg_approximation import approx_xg as _impl
    return _impl(elo_a, elo_d)


class StrengthsCache:
    """Cache incremental de strengths ponderados."""

    def __init__(
        self,
        df: pd.DataFrame,
        timeline: dict[str, dict[str, float]],
        elo_sigma: float = 225.0,
        recency_half_life_days: float = 1000.0,
        league_mean: float = 1.30,
        use_xg_real: bool = True,
    ) -> None:
        self.elo_sigma = elo_sigma
        self.recency_half_life_days = recency_half_life_days
        self.league_mean = league_mean
        self.timeline = timeline

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["home_goals", "away_goals"])
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)
        df = df.sort_values("date").reset_index(drop=True)
        self.df = df

        self.n = len(df)
        self.home_teams = df["home_team"].values.astype("U")
        self.away_teams = df["away_team"].values.astype("U")
        self.home_goals = df["home_goals"].values.astype(np.int32)
        self.away_goals = df["away_goals"].values.astype(np.int32)
        self.dates = df["date"].values

        # xG real lookup
        self._xg_real_lookup: dict = {}
        if use_xg_real:
            try:
                from src.data.statsbomb import get_xg_real_lookup
                self._xg_real_lookup = get_xg_real_lookup()
            except Exception:
                self._xg_real_lookup = {}

        # Team mapping
        all_teams = np.unique(np.concatenate([self.home_teams, self.away_teams]))
        self.team_to_idx = {t: i for i, t in enumerate(all_teams)}
        self.idx_to_team = all_teams
        self.n_teams = len(all_teams)

        # Agrupar partidos por fecha unica
        df_temp = pd.DataFrame({
            "date": self.dates,
            "orig_idx": np.arange(self.n),
        })
        grouped = df_temp.groupby("date")["orig_idx"].apply(list).sort_index()
        self._unique_dates = list(grouped.index)
        self._groups = [np.array(indices) for indices in grouped.values]

        # Estado
        self._state = np.zeros((self.n_teams, 6), dtype=np.float64)
        self._pos = 0
        self._elo_snapshot: dict[str, float] = {}
        self._snapshot_as_of: str | None = None

    def set_elo_snapshot(self, as_of: str) -> None:
        """Define el snapshot de Elo a usar. Recalcula xG y pesos de Elo.

        Idempotente: si el snapshot ya esta cargado para el mismo `as_of`,
        no reconstruye (evita el costo de 30s por llamada en scripts batch).
        Llamar una vez por Mundial (mismo elo_lookup para todos los partidos).
        """
        if self._snapshot_as_of == as_of and self._elo_snapshot:
            return
        self._snapshot_as_of = as_of
        self._elo_snapshot = get_elo_at(self.timeline, as_of)
        he = np.array([self._elo_snapshot.get(t, ORIGINAL_ELO) for t in self.home_teams])
        ae = np.array([self._elo_snapshot.get(t, ORIGINAL_ELO) for t in self.away_teams])

        # xG con el snapshot (mismo comportamiento que el original)
        self.xg_home = _approx_xg(he, ae)
        self.xg_away = _approx_xg(ae, he)

        # Sobrescribir con xG real de StatsBomb si esta disponible
        # Vectorizado: precomputar indice key->row y aplicar con numpy
        if self._xg_real_lookup:
            date_strs = pd.Series(self.dates).astype(str).str[:10].values
            # Construir mapping (date, home, away) -> row index
            row_idx: dict[tuple, int] = {}
            for i in range(self.n):
                row_idx[(date_strs[i], self.home_teams[i], self.away_teams[i])] = i
            # Aplicar todas las sobrescrituras en una sola pasada vectorizada
            indices = []
            real_h = []
            real_a = []
            for key, (xh, xa) in self._xg_real_lookup.items():
                if key in row_idx:
                    indices.append(row_idx[key])
                    real_h.append(xh)
                    real_a.append(xa)
            if indices:
                idx_arr = np.array(indices, dtype=np.int64)
                self.xg_home[idx_arr] = np.array(real_h, dtype=np.float64)
                self.xg_away[idx_arr] = np.array(real_a, dtype=np.float64)

        # Pesos de Elo con el snapshot
        elo_diff = (ae - he) / self.elo_sigma
        self.w_elo_home = np.exp(elo_diff).astype(np.float64)
        self.w_elo_away = np.exp(-elo_diff).astype(np.float64)

        # Reset state porque el xG cambio
        self._state[:] = 0.0
        self._pos = 0

    def advance_to(self, as_of: str | pd.Timestamp) -> pd.DataFrame:
        """Procesa todos los partidos con fecha < as_of y devuelve el state."""
        as_of_ts = pd.Timestamp(as_of)
        as_of_np = np.datetime64(as_of_ts)

        while self._pos < len(self._unique_dates):
            d = self._unique_dates[self._pos]
            if d >= as_of_np:
                break
            self._apply_group(self._pos, as_of_ts)
            self._pos += 1

        return self._state_df()

    def _apply_group(self, pos: int, ref_date: pd.Timestamp) -> None:
        indices = self._groups[pos]
        state = self._state

        ht_idx = np.array([self.team_to_idx[self.home_teams[i]] for i in indices])
        at_idx = np.array([self.team_to_idx[self.away_teams[i]] for i in indices])
        hg = self.home_goals[indices].astype(np.float64)
        ag = self.away_goals[indices].astype(np.float64)
        xgh = self.xg_home[indices]
        xga = self.xg_away[indices]

        dates_group = self.dates[indices]
        days_ago = (ref_date - pd.to_datetime(dates_group)).days.values.astype(np.float64)
        w_recency = 0.5 ** (days_ago / self.recency_half_life_days)
        wh = self.w_elo_home[indices] * w_recency
        wa = self.w_elo_away[indices] * w_recency

        np.add.at(state[:, 0], ht_idx, hg * wh)
        np.add.at(state[:, 1], ht_idx, ag * wh)
        np.add.at(state[:, 2], ht_idx, xgh * wh)
        np.add.at(state[:, 3], ht_idx, xga * wh)
        np.add.at(state[:, 4], ht_idx, wh)
        np.add.at(state[:, 5], ht_idx, 1.0)

        np.add.at(state[:, 0], at_idx, ag * wa)
        np.add.at(state[:, 1], at_idx, hg * wa)
        np.add.at(state[:, 2], at_idx, xga * wa)
        np.add.at(state[:, 3], at_idx, xgh * wa)
        np.add.at(state[:, 4], at_idx, wa)
        np.add.at(state[:, 5], at_idx, 1.0)

    def _state_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": self.idx_to_team,
            "gf_w": self._state[:, 0],
            "ga_w": self._state[:, 1],
            "xgf_w": self._state[:, 2],
            "xga_w": self._state[:, 3],
            "w_sum": self._state[:, 4],
            "matches": self._state[:, 5].astype(int),
        })

    def get_strengths(
        self,
        as_of: str,
        shrinkage_matches: int = 10,
        min_weighted_matches: float = 8.0,
    ) -> pd.DataFrame:
        raw = self.advance_to(as_of)
        mask = raw["w_sum"] >= min_weighted_matches
        valid = raw[mask].copy()
        if valid.empty:
            return valid

        xgf_w = np.maximum(valid["xgf_w"].values, 1e-9)
        xga_w = np.maximum(valid["xga_w"].values, 1e-9)
        attack = np.where(xgf_w > 0, valid["gf_w"].values / xgf_w, 1.0)
        defense = np.where(xga_w > 0, valid["ga_w"].values / xga_w, 1.0)

        w_sum = valid["w_sum"].values
        sf = w_sum / (w_sum + shrinkage_matches)
        attack = sf * attack + (1 - sf) * self.league_mean
        defense = sf * defense + (1 - sf) * self.league_mean

        return pd.DataFrame({
            "team": valid["team"].values,
            "attack": attack,
            "defense_vulnerability": defense,
            "matches": valid["matches"].values,
            "weighted_matches": w_sum,
        }).sort_values("attack", ascending=False).reset_index(drop=True)
