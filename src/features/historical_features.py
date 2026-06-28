"""Features historicas para ajustar strengths: H2H, momentum, WC history.

Cada feature devuelve multiplicadores (att_mult, def_mult) que se aplican
sobre los strengths base:

    adjusted_attack = base_attack * att_mult
    adjusted_def_vuln = base_defense_vuln * def_mult

att_mult > 1.0 => boost ataque
def_mult < 1.0 => mejora defensa (menos goles en contra esperados)

Magnitud maxima: ±10% (configurable via Settings mas adelante). Esto evita
que las features dominen la señal principal del strength historico.
"""
from __future__ import annotations

import pandas as pd

MAX_ADJUSTMENT = 0.10  # ±10% maximo ajuste por feature
MIN_H2H_MATCHES = 3    # minimo de H2H para que la feature aplique


def compute_h2h_adjustment(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of: str,
    n_matches: int = 5,
) -> tuple[float, float]:
    """Head-to-head: ajuste basado en ultimos N enfrentamientos entre los equipos.

    Para cada partido, contamos goles a favor y en contra de `home_team`.
    Si home_team ha dominado H2H (mas goles a favor que en contra), boost ataque
    y mejora defensa.

    Args:
        df: DataFrame con todos los partidos (columnas: date, home_team, away_team,
            home_goals, away_goals).
        home_team: nombre del equipo local.
        away_team: nombre del visitante.
        as_of: fecha de corte (str o Timestamp).
        n_matches: numero de H2H mas recientes a considerar.

    Returns:
        (att_mult, def_mult) con valores en [1-MAX, 1+MAX].
        Si no hay H2H suficientes, devuelve (1.0, 1.0).
    """
    as_of_ts = pd.Timestamp(as_of)
    mask = df["date"] < as_of_ts
    matches = df[mask & (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )].sort_values("date", ascending=False).head(n_matches)

    if len(matches) < MIN_H2H_MATCHES:
        return 1.0, 1.0

    # Calcular goles a favor / en contra de home_team en estos partidos
    gf = 0.0
    ga = 0.0
    for _, m in matches.iterrows():
        if m["home_team"] == home_team:
            gf += float(m["home_goals"])
            ga += float(m["away_goals"])
        else:  # home_team jugo de visitante
            gf += float(m["away_goals"])
            ga += float(m["home_goals"])

    n = len(matches)
    gf_per_game = gf / n
    ga_per_game = ga / n
    # League average (gol por partido por equipo) ~ 1.30
    league_avg = 1.30

    # Boost ataque proporcional a (gf_per_game - league_avg)
    att_boost = (gf_per_game - league_avg) / league_avg
    # Def mult: si concede menos, def_mult < 1 (mejor defensa)
    def_factor = (ga_per_game - league_avg) / league_avg

    # Clampear a MAX_ADJUSTMENT
    att_mult = 1.0 + max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, att_boost * MAX_ADJUSTMENT))
    def_mult = 1.0 + max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, def_factor * MAX_ADJUSTMENT))

    return float(att_mult), float(def_mult)


def compute_momentum_adjustment(
    df: pd.DataFrame,
    team: str,
    as_of: str,
    n_matches: int = 5,
    historical_avg_attack: float = 1.30,
    historical_avg_defense: float = 1.30,
) -> tuple[float, float]:
    """Momentum: ajuste basado en los ultimos N partidos del equipo.

    Args:
        df: DataFrame con partidos.
        team: nombre del equipo.
        as_of: fecha de corte.
        n_matches: ultimos N partidos del equipo.
        historical_avg_attack: avg de goles a favor (reference para comparar).
        historical_avg_defense: avg de goles en contra.

    Returns:
        (att_mult, def_mult) con valores en [1-MAX, 1+MAX].
        Si no hay partidos suficientes, (1.0, 1.0).
    """
    as_of_ts = pd.Timestamp(as_of)
    mask = df["date"] < as_of_ts
    matches = df[mask & (
        (df["home_team"] == team) | (df["away_team"] == team)
    )].sort_values("date", ascending=False).head(n_matches)

    if len(matches) < 3:
        return 1.0, 1.0

    gf = 0.0
    ga = 0.0
    for _, m in matches.iterrows():
        if m["home_team"] == team:
            gf += float(m["home_goals"])
            ga += float(m["away_goals"])
        else:
            gf += float(m["away_goals"])
            ga += float(m["home_goals"])

    n = len(matches)
    gf_per_game = gf / n
    ga_per_game = ga / n

    att_boost = (gf_per_game - historical_avg_attack) / historical_avg_attack
    def_factor = (ga_per_game - historical_avg_defense) / historical_avg_defense

    att_mult = 1.0 + max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, att_boost * MAX_ADJUSTMENT))
    def_mult = 1.0 + max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, def_factor * MAX_ADJUSTMENT))

    return float(att_mult), float(def_mult)


def compute_wc_history_adjustment(
    df: pd.DataFrame,
    team: str,
    as_of: str,
    tournament: str = "FIFA World Cup",
    min_wc_matches: int = 3,
) -> tuple[float, float]:
    """WC history: ajuste basado en el rendimiento historico en mundiales.

    Para un Mundial, los equipos con mas experiencia y mejor historial suelen
    rendir mejor. Equipos sin historial o con historial pobre reciben un
    pequeno descuento.

    Returns:
        (att_mult, def_mult) en [1-MAX, 1+MAX]. Default (1.0, 1.0).
    """
    as_of_ts = pd.Timestamp(as_of)
    mask = df["date"] < as_of_ts
    matches = df[mask & (df["tournament"] == tournament) & (
        (df["home_team"] == team) | (df["away_team"] == team)
    )]

    if len(matches) < min_wc_matches:
        return 1.0, 1.0

    wins = 0
    gf = 0.0
    ga = 0.0
    for _, m in matches.iterrows():
        hg, ag = float(m["home_goals"]), float(m["away_goals"])
        if m["home_team"] == team:
            gf += hg
            ga += ag
            if hg > ag:
                wins += 1
        else:
            gf += ag
            ga += hg
            if ag > hg:
                wins += 1

    n = len(matches)
    win_rate = wins / n  # 0.0 - 1.0
    # Equipo promedio: ~50% win rate. Boost linear con win_rate - 0.5.
    # Si win_rate = 0.7, att_mult = 1 + 0.2 * 0.10 = 1.02
    # Si win_rate = 0.3, att_mult = 1 - 0.2 * 0.10 = 0.98
    wc_factor = (win_rate - 0.5) * 2  # en [-1, 1]

    att_mult = 1.0 + max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, wc_factor * MAX_ADJUSTMENT))
    # Mejor win rate => mejor defensa => def_mult < 1
    def_mult = 1.0 - max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, wc_factor * MAX_ADJUSTMENT))

    return float(att_mult), float(def_mult)


def apply_all_adjustments(
    df: pd.DataFrame,
    home: str,
    away: str,
    as_of: str,
    enable_h2h: bool = True,
    enable_momentum: bool = True,
    enable_wc_history: bool = True,
) -> tuple[float, float, float, float]:
    """Combina todas las features en multiplicadores finales.

    Returns:
        (home_att_mult, home_def_mult, away_att_mult, away_def_mult).
    """
    h_att, h_def = 1.0, 1.0
    a_att, a_def = 1.0, 1.0

    if enable_h2h:
        h_att_h2h, h_def_h2h = compute_h2h_adjustment(df, home, away, as_of)
        a_att_h2h, a_def_h2h = compute_h2h_adjustment(df, away, home, as_of)
        h_att *= h_att_h2h
        h_def *= h_def_h2h
        a_att *= a_att_h2h
        a_def *= a_def_h2h

    if enable_momentum:
        h_att_m, h_def_m = compute_momentum_adjustment(df, home, as_of)
        a_att_m, a_def_m = compute_momentum_adjustment(df, away, as_of)
        h_att *= h_att_m
        h_def *= h_def_m
        a_att *= a_att_m
        a_def *= a_def_m

    if enable_wc_history:
        h_att_w, h_def_w = compute_wc_history_adjustment(df, home, as_of)
        a_att_w, a_def_w = compute_wc_history_adjustment(df, away, as_of)
        h_att *= h_att_w
        h_def *= h_def_w
        a_att *= a_att_w
        a_def *= a_def_w

    return h_att, h_def, a_att, a_def


def compute_match_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of: str,
    enable: bool = True,
) -> tuple[float, float, float, float]:
    """Wrapper one-liner para usar en predict pipelines.

    Args:
        df: DataFrame con todos los partidos.
        home_team, away_team: nombres (martj).
        as_of: fecha de corte.
        enable: si False, devuelve (1, 1, 1, 1) (no adjustment).

    Returns:
        (home_att_mult, home_def_mult, away_att_mult, away_def_mult).
    """
    if not enable:
        return 1.0, 1.0, 1.0, 1.0
    return apply_all_adjustments(df, home_team, away_team, as_of)
