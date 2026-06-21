"""Llave completa de eliminatorias basada en N simulaciones Monte Carlo.

Para cada partido de la llave (R32, R16, QF, SF, Final), calcula:
- % de veces que cada equipo juega ese partido (appearances)
- % de veces que cada equipo gana ese partido (wins)

Uso:
    from src.simulation.bracket_analysis import analyze_bracket
    analysis = analyze_bracket(sim, fixtures, n_simulations=1000)
    print(analysis["r32"][73])  # partido 73 (R32)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.simulation.wc2026_bracket import (
    FINAL,
    QUARTER_FINALS,
    ROUND_OF_16,
    ROUND_OF_32,
    SEMI_FINALS,
    SlotKind,
)
from src.simulation.wc2026_simulate import TournamentSimulator, simulate_tournament


@dataclass
class MatchAnalysis:
    """Análisis de un partido de la llave a través de N simulaciones."""

    tie_id: int
    round_name: str  # "R32", "R16", "QF", "SF", "F"
    home_label: str  # "Winner A", "Runner B", "3rd {A,B,C,D,F}", "Wo73", etc.
    away_label: str
    appearances: Counter = field(default_factory=Counter)
    wins: Counter = field(default_factory=Counter)
    n_simulations: int = 0

    @property
    def appearances_pct(self) -> dict[str, float]:
        if self.n_simulations == 0:
            return {}
        return {
            team: count / self.n_simulations
            for team, count in self.appearances.items()
        }

    @property
    def wins_pct(self) -> dict[str, float]:
        if self.n_simulations == 0:
            return {}
        return {
            team: count / self.n_simulations
            for team, count in self.wins.items()
        }


def _slot_label(slot) -> str:
    """Genera etiqueta legible para un slot del bracket."""
    if slot.kind == SlotKind.GROUP_WINNER:
        return f"Winner {slot.group}"
    if slot.kind == SlotKind.GROUP_RUNNER_UP:
        return f"Runner {slot.group}"
    if slot.kind == SlotKind.GROUP_THIRD:
        opts = ",".join(slot.third_options)
        return f"3rd {{{opts}}}"
    if slot.kind == SlotKind.WINNER_OF:
        return f"Wo{slot.tie_id}"
    return "?"


def _build_match_analyses(round_name: str, ties) -> dict[int, MatchAnalysis]:
    """Inicializa el dict de MatchAnalysis para una ronda."""
    return {
        tie.id: MatchAnalysis(
            tie_id=tie.id,
            round_name=round_name,
            home_label=_slot_label(tie.home),
            away_label=_slot_label(tie.away),
        )
        for tie in ties
    }


def _resolve_team(slot, result) -> str | None:
    """Resuelve un slot a un team name concreto dado el resultado."""
    if slot.kind == SlotKind.GROUP_WINNER:
        return result.get("group_winners", {}).get(slot.group)
    if slot.kind == SlotKind.GROUP_RUNNER_UP:
        return result.get("group_runners_up", {}).get(slot.group)
    if slot.kind == SlotKind.GROUP_THIRD:
        # No podemos saber que grupo especifico fue asignado a este slot
        # sin rastrear la asignacion. Devolvemos None.
        return None
    if slot.kind == SlotKind.WINNER_OF:
        return result.get("winners", {}).get(slot.tie_id)
    return None


def analyze_bracket(
    sim: TournamentSimulator,
    fixtures: pd.DataFrame,
    n_simulations: int = 1000,
) -> dict:
    """Corre N simulaciones y analiza la probabilidad de cada partido de la llave.

    Para appearances, hace un "playback": despues de la simulacion, rastrea
    los slots para encontrar que equipos especificos jugaron cada partido.
    Para third_place slots, usa el team que aparece en winners[] del match
    R32 (si un equipo gano, es porque jugo, y podemos atribuir su participation
    a la union de los slots donde pudo haber sido asignado).
    """
    # Inicializar analyses
    analyses_r32 = _build_match_analyses("R32", ROUND_OF_32)
    analyses_r16 = _build_match_analyses("R16", ROUND_OF_16)
    analyses_qf = _build_match_analyses("QF", QUARTER_FINALS)
    analyses_sf = _build_match_analyses("SF", SEMI_FINALS)
    analyses_f = {FINAL.id: MatchAnalysis(
        tie_id=FINAL.id,
        round_name="Final",
        home_label="Wo101",
        away_label="Wo102",
    )}

    all_analyses: dict[int, MatchAnalysis] = {}
    for rnd in [analyses_r32, analyses_r16, analyses_qf, analyses_sf, analyses_f]:
        for tid, ma in rnd.items():
            ma.n_simulations = n_simulations
            all_analyses[tid] = ma

    n_errors = 0
    for i in range(n_simulations):
        rng = np.random.default_rng(42 + i)
        result = simulate_tournament(sim, fixtures, rng)
        if "error" in result:
            n_errors += 1
            continue

        # Para cada partido, registrar wins (equipo concreto que gano)
        for tid, winner in result.get("winners", {}).items():
            if tid in all_analyses:
                all_analyses[tid].wins[winner] += 1

        # Para appearances: resolver slots
        for round_ties in [ROUND_OF_32, ROUND_OF_16, QUARTER_FINALS, SEMI_FINALS, [FINAL]]:
            for tie in round_ties:
                ma = all_analyses.get(tie.id)
                if ma is None:
                    continue
                home_team = _resolve_team(tie.home, result)
                away_team = _resolve_team(tie.away, result)
                if home_team and home_team != "?":
                    ma.appearances[home_team] += 1
                if away_team and away_team != "?":
                    ma.appearances[away_team] += 1

        # Special: para slots GROUP_THIRD, el equipo concreto lo sabemos por
        # los winners de R32. Un equipo que gana un R32 es un 3rd qualified.
        # Asignamos su appearance al R32 donde aparece.
        for tie in ROUND_OF_32:
            ma = all_analyses.get(tie.id)
            if ma is None:
                continue
            winner = result.get("winners", {}).get(tie.id)
            if winner is None:
                continue
            # Si winner es 3rd de algun grupo (i.e., aparece en third_teams)
            is_third_team = winner in result.get("third_teams", {}).values()
            if not is_third_team:
                continue
            # Atribuir appearance a ESTE partido (ya fue contado arriba
            # si el slot resolvio, pero como slot third no resuelve,
            # ahora lo sumamos).
            if tie.home.kind == SlotKind.GROUP_THIRD and winner not in ma.appearances:
                ma.appearances[winner] += 1
            if tie.away.kind == SlotKind.GROUP_THIRD and winner not in ma.appearances:
                ma.appearances[winner] += 1

    return {
        "r32": analyses_r32,
        "r16": analyses_r16,
        "qf": analyses_qf,
        "sf": analyses_sf,
        "final": analyses_f,
        "n_simulations": n_simulations,
        "n_errors": n_errors,
    }


def format_bracket_table(analysis: dict) -> str:
    """Formatea el analysis como tabla Markdown."""
    lines = []
    n = analysis["n_simulations"]

    lines.append(f"## Llave completa de eliminatorias ({n} simulaciones MC)")
    lines.append("")
    lines.append(
        f"Para cada partido, mostramos la probabilidad de que cada equipo "
        f"juegue (**Juega**) y gane (**Gana**) ese partido, "
        f"agregado sobre {n} simulaciones."
    )
    lines.append("")
    lines.append(
        "*Nota: para partidos de R32 con un slot de 3rd "
        "(ej. 'Winner E vs 3rd {A,B,C,D,F}'), "
        "la columna 'Juega' puede no atribuirse al 3rd correcto por ambiguedad "
        "del bracket, pero la columna 'Gana' es exacta.*"
    )
    lines.append("")

    rounds = [
        ("R32", analysis["r32"]),
        ("R16", analysis["r16"]),
        ("QF", analysis["qf"]),
        ("SF", analysis["sf"]),
        ("Final", analysis["final"]),
    ]

    for rnd_name, rnd_analyses in rounds:
        lines.append(f"### {rnd_name}")
        lines.append("")
        for tid in sorted(rnd_analyses.keys()):
            ma = rnd_analyses[tid]
            lines.append(f"**Partido {tid}** ({ma.home_label} vs {ma.away_label})")
            lines.append("")
            all_teams = set(ma.appearances) | set(ma.wins)
            if not all_teams:
                lines.append("- (sin datos)")
                lines.append("")
                continue
            rows = []
            for team in all_teams:
                app = ma.appearances.get(team, 0)
                win = ma.wins.get(team, 0)
                rows.append((team, app / n, win / n))
            rows.sort(key=lambda r: (-r[1], -r[2]))
            lines.append("| Equipo | Juega | Gana |")
            lines.append("|---|---:|---:|")
            for team, app_pct, win_pct in rows:
                lines.append(f"| {team} | {app_pct:.1%} | {win_pct:.1%} |")
            lines.append("")

    return "\n".join(lines)
