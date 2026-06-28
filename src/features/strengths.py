"""Cálculo de attack/defense strength ponderado por calidad del rival.

Problema: si Sweden promedia goles contra cualquier rival, le infla el
ataque. Queremos que el ataque de un equipo refleje su capacidad ofensiva
*contra rivales comparables*.

Solución: para cada partido donde el equipo jugó, calculamos un peso en
función de la diferencia de Elo con el rival. Goles contra rivales
fuertes pesan más, goles contra rivales débiles pesan menos.

Peso: w = exp(-|elo_diff| / sigma), con sigma = 200 por defecto.
Esto significa que un rival con Elo 200 puntos menor tiene peso ~0.37,
y un rival con Elo 200 puntos mayor tiene peso ~2.7.

Adicionalmente:
- Recencia: partidos más recientes pesan más (decay exponencial)
- Mínimo de partidos: si el equipo tiene < N partidos, hacemos shrinkage
  hacia la media de la liga.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.data.elo import ORIGINAL_ELO


class WeightedTeamStrength(BaseModel):
    """Strength ponderado de un equipo."""

    name: str
    attack: float
    defense_vulnerability: float
    matches: int = 0
    weighted_matches: float = 0.0
    attack_std: float | None = None
    defense_std: float | None = None


def compute_weighted_strengths(
    df: pd.DataFrame,
    elo_lookup: dict[str, float],
    as_of: str | None = None,
    elo_sigma: float = 200.0,
    recency_half_life_days: float = 730.0,
    shrinkage_matches: int = 10,
    league_mean_attack: float = 1.30,
    league_mean_defense: float = 1.30,
    min_weighted_matches: float = 5.0,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
    use_xg_real: bool = True,
) -> pd.DataFrame:
    """Calcula attack/defense strength ponderado por Elo del rival y recencia.

    Args:
        df: DataFrame con partidos históricos.
        elo_lookup: dict team -> Elo. Se usa el Elo del RIVAL en cada partido
            (el del equipo target se asume implícito en su match history).
        as_of: fecha de corte. Partidos después de esta se ignoran.
            Si None, usa la última fecha del dataset.
        elo_sigma: parámetro de la exponencial de peso por Elo.
            Mayor = más permisivo con diferencias de Elo.
        recency_half_life_days: vida media del peso por recencia.
        shrinkage_matches: número de partidos para shrinkage bayesiano.
        league_mean_attack/defense: priors para shrinkage.
        min_weighted_matches: mínimo de partidos ponderados para incluir
            al equipo en el output.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[home_goals_col, away_goals_col])
    df[home_goals_col] = df[home_goals_col].astype(int)
    df[away_goals_col] = df[away_goals_col].astype(int)

    if as_of is not None:
        as_of_dt = pd.Timestamp(as_of)
        df = df[df[date_col] <= as_of_dt]
    if df.empty:
        return pd.DataFrame()

    ref_date = df[date_col].max()

    # Vectorized: calcular todo en numpy/pandas, mucho más rápido
    home_elo_arr = df[home_col].map(lambda t: elo_lookup.get(t, ORIGINAL_ELO)).values
    away_elo_arr = df[away_col].map(lambda t: elo_lookup.get(t, ORIGINAL_ELO)).values

    elo_diff_home = (away_elo_arr - home_elo_arr) / elo_sigma
    elo_diff_away = -elo_diff_home
    w_for_home = np.exp(elo_diff_home)
    w_for_away = np.exp(elo_diff_away)

    days_ago = (ref_date - df[date_col]).dt.days.values
    w_recency = 0.5 ** (days_ago / recency_half_life_days)

    # xG: intentar usar real de StatsBomb, fallback a aproximacion por Elo
    xg_home = _approx_xg_from_elo(home_elo_arr, away_elo_arr)
    xg_away = _approx_xg_from_elo(away_elo_arr, home_elo_arr)
    if use_xg_real:
        from src.data.statsbomb import get_xg_real
        n = len(df)
        home_names = df[home_col].values
        away_names = df[away_col].values
        dates = df[date_col].dt.strftime("%Y-%m-%d").values
        real_count = 0
        for i in range(n):
            xg = get_xg_real(dates[i], home_names[i], away_names[i])
            if xg is not None:
                xg_home[i] = xg[0]
                xg_away[i] = xg[1]
                real_count += 1
        if real_count > 0:
            # Log solo la primera vez por sesion
            import os
            if not os.environ.get("_XGBOMB_LOGGED"):
                print(f"  xG real usado en {real_count}/{n} partidos previos", flush=True)
                os.environ["_XGBOMB_LOGGED"] = "1"

    # Construir DataFrame con filas duplicadas (home y away por separado)
    n = len(df)
    teams = np.concatenate([df[home_col].values, df[away_col].values])
    gf = np.concatenate([df[home_goals_col].values, df[away_goals_col].values])
    ga = np.concatenate([df[away_goals_col].values, df[home_goals_col].values])
    xg_for = np.concatenate([xg_home, xg_away])
    xg_against = np.concatenate([xg_away, xg_home])
    weights = np.concatenate([w_for_home * w_recency, w_for_away * w_recency])

    per_match = pd.DataFrame({
        "team": teams,
        "gf": gf,
        "ga": ga,
        "xg_for": xg_for,
        "xg_against": xg_against,
        "weight": weights,
    })

    # Para cada equipo, calcular attack/defense como relación goles/xG ponderada
    # attack = sum(gf * w) / sum(xg_for * w)
    # defense_vulnerability = sum(ga * w) / sum(xg_against * w)
    grouped = per_match.groupby("team").apply(_aggregate_team).reset_index()

    # Shrinkage bayesiano hacia la media
    grouped["weighted_n"] = grouped["weighted_matches"]
    grouped = grouped[grouped["weighted_n"] >= min_weighted_matches].copy()

    if not grouped.empty:
        # Factor de shrinkage: (n / (n + k)) * valor + (k / (n + k)) * media
        shrink_factor = grouped["weighted_n"] / (grouped["weighted_n"] + shrinkage_matches)
        grouped["attack"] = (
            shrink_factor * grouped["attack"] + (1 - shrink_factor) * league_mean_attack
        )
        grouped["defense_vulnerability"] = (
            shrink_factor * grouped["defense_vulnerability"]
            + (1 - shrink_factor) * league_mean_defense
        )

    return grouped.sort_values("attack", ascending=False).reset_index(drop=True)


def _aggregate_team(group: pd.DataFrame) -> pd.Series:
    """Agrega goles/xG ponderados por equipo."""
    w = group["weight"]
    attack = (group["gf"] * w).sum() / (group["xg_for"] * w).sum() if (group["xg_for"] * w).sum() > 0 else 1.0
    defense = (group["ga"] * w).sum() / (group["xg_against"] * w).sum() if (group["xg_against"] * w).sum() > 0 else 1.0

    # Bootstrap-style std estimation (sample variance of gf per match)
    if len(group) > 1:
        gf_per_match = group["gf"].values
        attack_std = float(np.std(gf_per_match, ddof=1))
        defense_std = float(np.std(group["ga"].values, ddof=1))
    else:
        attack_std = None
        defense_std = None

    return pd.Series({
        "attack": attack,
        "defense_vulnerability": defense,
        "matches": len(group),
        "weighted_matches": w.sum(),
        "attack_std": attack_std,
        "defense_std": defense_std,
    })


def _approx_xg_from_elo(elo_attacker: float, elo_defender: float) -> float:
    """DEPRECATED: usa src.features.xg_approximation.approx_xg_from_elo.

    Aproxima xG esperado desde la diferencia de Elo. Mantenido como shim.
    """
    from src.features.xg_approximation import approx_xg_from_elo as _impl
    return _impl(elo_attacker, elo_defender)


def build_elo_lookup_at(
    elo_system,
    as_of_date: str,
) -> dict[str, float]:
    """Construye un lookup de Elo al momento de una fecha dada.

    Recorre el historial del elo_system y devuelve el último rating
    conocido de cada equipo hasta as_of_date.
    """
    lookup: dict[str, float] = {}
    for u in elo_system.history:
        if u.date and u.date <= as_of_date:
            lookup[u.home_team] = u.home_elo_after
            lookup[u.away_team] = u.away_elo_after
    # Inicializar equipos sin historial
    for team, r in elo_system.ratings.items():
        if team not in lookup:
            lookup[team] = r
    return lookup
