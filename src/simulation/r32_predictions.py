"""Calcular bracket R32 del WC 2026 a partir de resultados reales.

Pasos:
1. Calcular standings de cada grupo (pts, GD, GF)
2. Determinar 8 mejores terceros
3. Asignar terceros a slots del bracket
4. Listar los 16 cruces de R32 con predicciones del modelo

Nota: las predicciones se pasan a `predict_fn(home, away)` que es una
función inyectada para evitar acoplamiento con TournamentSimulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.data.wc2026_fixture import GROUPS
from src.simulation.wc2026_bracket import ROUND_OF_32, SlotKind


@dataclass
class R32Match:
    tie_id: int
    home_team: str
    away_team: str
    home_source: str  # descripcion humana: "Winner A", "3rd K/L", etc.
    away_source: str
    p_home: float
    p_draw: float
    p_away: float
    most_likely: tuple[int, int]


def _olo_name_to_martj(olo: str, olo_to_martj: dict) -> str:
    return olo_to_martj.get(olo, olo)


def compute_group_standings(fixtures: pd.DataFrame) -> dict[str, list[tuple[str, int, int, int]]]:
    """Calcula standings (ordenados) de cada grupo.

    Returns:
        dict[group, list of (team, pts, gd, gf)] ordenados por:
        1. pts (desc), 2. gd (desc), 3. gf (desc)

    Nota: deriva los nombres de equipo de los fixtures (columna `home`)
    para evitar inconsistencias entre GROUPS y los nombres OLO reales.
    """
    standings = {}
    for group in GROUPS:
        group_fx = fixtures[fixtures["group"] == group]
        teams_in_group = sorted(
            {fx["home"] for _, fx in group_fx.iterrows()}
            | {fx["away"] for _, fx in group_fx.iterrows()}
        )
        pts = {t: 0 for t in teams_in_group}
        gf = {t: 0 for t in teams_in_group}
        ga = {t: 0 for t in teams_in_group}
        for _, fx in group_fx.iterrows():
            if not fx["played"]:
                continue
            h, a = fx["home"], fx["away"]
            hg, ag = int(fx["home_score"]), int(fx["away_score"])
            pts[h] += 3 if hg > ag else (1 if hg == ag else 0)
            pts[a] += 3 if ag > hg else (1 if hg == ag else 0)
            gf[h] += hg
            ga[h] += ag
            gf[a] += ag
            ga[a] += hg

        sorted_teams = sorted(
            teams_in_group, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t])
        )
        standings[group] = [
            (t, pts[t], gf[t] - ga[t], gf[t]) for t in sorted_teams
        ]
    return standings


def get_top_8_thirds(standings: dict) -> list[tuple[str, str, int, int, int]]:
    """Retorna los 8 mejores terceros ordenados por pts, GD, GF.

    Returns:
        list of (group, team, pts, gd, gf)
    """
    thirds = []
    for group, table in standings.items():
        if len(table) >= 3:
            t, pts, gd, gf = table[2]
            thirds.append((group, t, pts, gd, gf))
    thirds.sort(key=lambda x: (-x[2], -x[3], -x[4]))
    return thirds[:8]


def _slot_to_label(slot) -> str:
    """Convierte un slot del bracket a texto humano."""
    if slot.kind == SlotKind.GROUP_WINNER:
        return f"Winner {slot.group}"
    if slot.kind == SlotKind.GROUP_RUNNER_UP:
        return f"Runner {slot.group}"
    if slot.kind == SlotKind.GROUP_THIRD:
        groups_str = "/".join(slot.third_options)
        return f"3rd ({groups_str})"
    if slot.kind == SlotKind.WINNER_OF:
        return f"Winner of #{slot.tie_id}"
    return "?"


def build_r32_matches(
    fixtures: pd.DataFrame,
    olo_to_martj: dict,
    predict_fn: Callable[[str, str], dict],
) -> list[R32Match]:
    """Construye la lista de 16 partidos de R32 con predicciones.

    Args:
        fixtures: DataFrame con todos los partidos de grupo (72 filas, todos played).
        olo_to_martj: mapping de nombre OLO -> nombre martj42.
        predict_fn: callable(home_olo, away_olo) -> dict con keys
            "p_h", "p_d", "p_a", "most_likely".
    """
    standings = compute_group_standings(fixtures)
    group_winners = {g: t[0][0] for g, t in standings.items()}
    group_runners_up = {g: t[1][0] for g, t in standings.items()}
    top_8_thirds = get_top_8_thirds(standings)
    third_teams = {g: (team, pts, gd, gf) for g, team, pts, gd, gf in top_8_thirds}
    qualified_third_groups = [g for g, *_ in top_8_thirds]

    # Asignar terceros a slots
    from src.simulation.wc2026_bracket import assign_third_place_slots
    third_assignments = assign_third_place_slots(qualified_third_groups)
    if third_assignments is None:
        raise RuntimeError(
            f"No se pudo asignar terceros: {qualified_third_groups}"
        )

    matches = []
    for tie in ROUND_OF_32:
        # Resolver slots
        def resolve(slot):
            if slot.kind == SlotKind.GROUP_WINNER:
                return group_winners[slot.group], _slot_to_label(slot)
            if slot.kind == SlotKind.GROUP_RUNNER_UP:
                return group_runners_up[slot.group], _slot_to_label(slot)
            if slot.kind == SlotKind.GROUP_THIRD:
                g = third_assignments[tie.id]
                return third_teams[g][0], _slot_to_label(slot)
            return "?", _slot_to_label(slot)

        home, home_label = resolve(tie.home)
        away, away_label = resolve(tie.away)
        pred = predict_fn(home, away)
        matches.append(
            R32Match(
                tie_id=tie.id,
                home_team=home,
                away_team=away,
                home_source=home_label,
                away_source=away_label,
                p_home=pred["p_h"],
                p_draw=pred["p_d"],
                p_away=pred["p_a"],
                most_likely=pred["most_likely"],
            )
        )
    return matches, standings, top_8_thirds


def format_standings_table(standings: dict) -> str:
    """Formatea standings como tabla Markdown."""
    lines = []
    for group in sorted(standings.keys()):
        lines.append(f"### Group {group}\n")
        lines.append("| Pos | Equipo | PTS | GD | GF |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, (team, pts, gd, gf) in enumerate(standings[group], 1):
            gd_str = f"+{gd}" if gd > 0 else str(gd)
            lines.append(f"| {i} | {team} | {pts} | {gd_str} | {gf} |")
        lines.append("")
    return "\n".join(lines)


def format_r32_table(matches: list[R32Match]) -> str:
    """Formatea los 16 partidos R32 como tabla Markdown."""
    lines = [
        "| # | Home | H% | D% | A% | Away | Marcador |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for m in matches:
        ml = f"{m.most_likely[0]}-{m.most_likely[1]}"
        lines.append(
            f"| #{m.tie_id} | **{m.home_team}** | {m.p_home:.0%} | "
            f"{m.p_draw:.0%} | {m.p_away:.0%} | {m.away_team} | {ml} |"
        )
    return "\n".join(lines)
