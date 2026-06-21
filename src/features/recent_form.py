"""Forma reciente: attack/defense basada solo en los últimos N partidos
de cada equipo, con peso exponencial opcional.

Se combina con el modelo base via:
    final_attack = w_recent * recent_attack + (1 - w_recent) * historical_attack

El histórico viene de compute_weighted_strengths; el reciente de
compute_recent_form. Esto captura rachas (equipos en forma / fuera de forma)
que el promedio ponderado diluye.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_recent_form(
    df: pd.DataFrame,
    as_of: str | None = None,
    n_matches: int = 5,
    decay_half_life_matches: float = 3.0,
    league_mean: float = 1.30,
    min_matches: int = 3,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
) -> pd.DataFrame:
    """Calcula attack/defense de cada equipo en sus últimos N partidos.

    Para cada equipo, toma los últimos `n_matches` partidos jugados
    (ordenados por fecha descendente) y calcula:
        attack = mean(gf de los últimos N) / league_mean
        defense_vulnerability = mean(ga de los últimos N) / league_mean

    Opcionalmente aplica decay exponencial por partido (más reciente pesa más).

    Args:
        df: DataFrame con partidos.
        as_of: fecha de corte (string o datetime).
        n_matches: número de últimos partidos a considerar.
        decay_half_life_matches: vida media del peso por orden cronológico.
            None o 0 desactiva el decay.
        league_mean: prior para shrinkage.
        min_matches: mínimo de partidos para incluir al equipo.

    Returns:
        DataFrame con columnas: team, recent_attack, recent_defense,
        recent_matches, recent_weighted_matches.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[home_goals_col, away_goals_col])

    if as_of is not None:
        as_of_dt = pd.Timestamp(as_of)
        df = df[df[date_col] <= as_of_dt]
    if df.empty:
        return pd.DataFrame()

    # Expandir: una fila por (equipo, partido) con su gf/ga
    home = df[[date_col, home_col, home_goals_col, away_goals_col]].rename(
        columns={home_col: "team", home_goals_col: "gf", away_goals_col: "ga"}
    )
    away = df[[date_col, away_col, away_goals_col, home_goals_col]].rename(
        columns={away_col: "team", away_goals_col: "gf", home_goals_col: "ga"}
    )
    long = pd.concat([home, away], ignore_index=True).sort_values([date_col, "team"])

    # Para cada equipo, tomar los últimos N partidos
    results = []
    for team, group in long.groupby("team"):
        last_n = group.tail(n_matches)
        if len(last_n) < min_matches:
            continue
        gf = last_n["gf"].values
        ga = last_n["ga"].values
        if decay_half_life_matches and decay_half_life_matches > 0:
            # El último partido tiene peso 1, el anterior tiene peso 0.5^(1/halflife), etc.
            # Invertimos para que el más reciente tenga índice 0
            n = len(gf)
            weights = np.power(0.5, np.arange(n)[::-1] / decay_half_life_matches)
            weights /= weights.sum()
            w_att = (gf * weights).sum()
            w_def = (ga * weights).sum()
        else:
            w_att = gf.mean()
            w_def = ga.mean()

        results.append({
            "team": team,
            "recent_attack": w_att / league_mean,
            "recent_defense": w_def / league_mean,
            "recent_matches": len(last_n),
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    # Shrinkage hacia la media (más agresivo para recent form por tener menos datos)
    shrink_k = 3.0
    sf = out["recent_matches"] / (out["recent_matches"] + shrink_k)
    out["recent_attack"] = sf * out["recent_attack"] + (1 - sf) * 1.0
    out["recent_defense"] = sf * out["recent_defense"] + (1 - sf) * 1.0

    return out


def blend_recent_with_historical(
    historical: pd.DataFrame,
    recent: pd.DataFrame,
    weight_recent: float = 0.0,
) -> pd.DataFrame:
    """Combina attack/defense histórico con reciente.

    Args:
        historical: DataFrame con team, attack, defense_vulnerability.
        recent: DataFrame con team, recent_attack, recent_defense.
        weight_recent: 0.0 = solo histórico, 1.0 = solo reciente.

    Returns:
        DataFrame con team, attack, defense_vulnerability (mezclados).
    """
    if historical.empty:
        return historical
    if recent.empty or weight_recent <= 0:
        return historical

    out = historical.merge(
        recent[["team", "recent_attack", "recent_defense"]],
        on="team", how="left",
    )
    has_recent = out["recent_attack"].notna()
    out.loc[has_recent, "attack"] = (
        weight_recent * out.loc[has_recent, "recent_attack"]
        + (1 - weight_recent) * out.loc[has_recent, "attack"]
    )
    out.loc[has_recent, "defense_vulnerability"] = (
        weight_recent * out.loc[has_recent, "recent_defense"]
        + (1 - weight_recent) * out.loc[has_recent, "defense_vulnerability"]
    )
    return out.drop(columns=["recent_attack", "recent_defense"])
