"""Llave completa de eliminatorias basada en N simulaciones Monte Carlo.

Para cada partido de la llave (R32, R16, QF, SF, Final), calcula:
- Probabilidad de que un equipo sea el "favorito" para GANAR ese partido
  (basado en la fraccion de simulaciones donde ese equipo gano).

Salida: una llave compacta en formato markdown donde cada partido muestra
los 2-3 equipos mas probables de ganarlo (no la lista exhaustiva).

Uso:
    from src.simulation.bracket_analysis import analyze_bracket
    analysis = analyze_bracket(sim, fixtures, n_simulations=1000)
    md = format_bracket_tree(analysis)
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
from src.simulation.wc2026_simulate import TournamentSimulator


@dataclass
class MatchAnalysis:
    """Analisis de un partido de la llave a traves de N simulaciones."""

    tie_id: int
    round_name: str
    home_label: str
    away_label: str
    wins: Counter = field(default_factory=Counter)
    n_simulations: int = 0

    @property
    def wins_pct(self) -> dict[str, float]:
        if self.n_simulations == 0:
            return {}
        return {
            team: count / self.n_simulations
            for team, count in self.wins.items()
        }

    def top_favorites(self, k: int = 3) -> list[tuple[str, float]]:
        """Devuelve los k equipos mas probables de ganar este partido."""
        return sorted(
            self.wins_pct.items(),
            key=lambda x: -x[1],
        )[:k]


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
    return {
        tie.id: MatchAnalysis(
            tie_id=tie.id,
            round_name=round_name,
            home_label=_slot_label(tie.home),
            away_label=_slot_label(tie.away),
        )
        for tie in ties
    }


def analyze_bracket(
    sim: TournamentSimulator,
    fixtures: pd.DataFrame,
    n_simulations: int = 1000,
) -> dict:
    """Corre N simulaciones y agrega los ganadores de cada partido de la llave.

    Para cada tie de la llave, cuenta que equipo gano en cada simulacion.
    El resultado es un dict[tie_id, MatchAnalysis].
    """
    from src.simulation.wc2026_simulate import simulate_tournament

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

        for tid, winner in result.get("winners", {}).items():
            if tid in all_analyses:
                all_analyses[tid].wins[winner] += 1

    return {
        "r32": analyses_r32,
        "r16": analyses_r16,
        "qf": analyses_qf,
        "sf": analyses_sf,
        "final": analyses_f,
        "n_simulations": n_simulations,
        "n_errors": n_errors,
    }


def format_bracket_tree(analysis: dict) -> str:
    """Formatea la llave como tabla compacta por partido.

    Para cada partido, muestra:
    - Slot home/away (que tipo de equipo juega ahi: Winner A, Runner B, etc.)
    - Top 3 favoritos para GANAR ese partido especifico
    """
    n = analysis["n_simulations"]
    lines = []
    lines.append(f"## Llave de eliminatorias ({n} simulaciones)")
    lines.append("")
    lines.append(
        "Para cada partido, los **3 favoritos para GANAR** ese partido "
        "(probabilidad condicional de ganar ese match especifico, no de campeon)."
    )
    lines.append("")

    rounds = [
        ("Round of 32", analysis["r32"]),
        ("Round of 16", analysis["r16"]),
        ("Quarterfinals", analysis["qf"]),
        ("Semifinals", analysis["sf"]),
        ("Final", analysis["final"]),
    ]

    for rnd_name, rnd_analyses in rounds:
        lines.append(f"### {rnd_name}")
        lines.append("")
        for tid in sorted(rnd_analyses.keys()):
            ma = rnd_analyses[tid]
            favs = ma.top_favorites(k=3)
            home = ma.home_label
            away = ma.away_label
            lines.append(f"**#{tid}**  {home}  vs  {away}")
            lines.append("")
            if not favs:
                lines.append("- (sin datos)")
                lines.append("")
                continue
            # Solo mostrar favoritos con prob > 1% (filtrar ruido)
            favs_filtered = [(t, p) for t, p in favs if p > 0.01]
            if not favs_filtered:
                favs_filtered = favs[:1]
            for team, prob in favs_filtered:
                lines.append(f"- **{team}** {prob:.0%}")
            lines.append("")

    return "\n".join(lines)
