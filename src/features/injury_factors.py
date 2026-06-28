"""Calculo de multiplicadores por lesiones/suspendidos (unica implementacion).

Consolidacion: existia duplicado en src/cli/wc2026_readme.py y
src/simulation/wc2026_simulate.py. Tambien existia una version
incompatible en src/data/injuries.py (TeamInjuries.attack_penalty) que
no se usaba.

Penalizaciones conservadoras: un jugador top no elimina mas del 20%
del equipo, porque hay suplentes que cubren parcialmente. Configurables
via Settings.
"""
from __future__ import annotations

from src.config import get_settings


def injury_factors(injuries: dict | None, team_martj: str) -> tuple[float, float]:
    """Calcula multiplicadores (attack, defense) basados en lesionados.

    Args:
        injuries: dict[team_martj, TeamInjuries]. Si None o equipo ausente, retorna (1.0, 1.0).
        team_martj: nombre del equipo en formato martj42.

    Returns:
        (attack_mult, defense_mult) donde 1.0 = sin ajuste.

    Penalizaciones (configurables via Settings):
        - OUT: max 20% reduccion de attack, max 15% aumento de vulnerability
        - DOUBTFUL: 50% del penalty de OUT
    """
    if not injuries:
        return 1.0, 1.0
    ti = injuries.get(team_martj)
    if ti is None or (not ti.out and not ti.doubtful):
        return 1.0, 1.0

    s = get_settings()
    out_attack = min(
        s.injury_max_attack_penalty,
        sum(p.importance for p in ti.out if p.position in ("FWD", "MID")) * s.injury_max_attack_penalty,
    )
    out_defense = min(
        s.injury_max_defense_penalty,
        sum(p.importance for p in ti.out if p.position in ("DEF", "GK")) * s.injury_max_defense_penalty,
    )
    dout_attack = min(
        s.injury_max_attack_penalty * s.injury_doubtful_factor,
        sum(p.importance for p in ti.doubtful if p.position in ("FWD", "MID")) * s.injury_max_attack_penalty * s.injury_doubtful_factor,
    )
    dout_defense = min(
        s.injury_max_defense_penalty * s.injury_doubtful_factor,
        sum(p.importance for p in ti.doubtful if p.position in ("DEF", "GK")) * s.injury_max_defense_penalty * s.injury_doubtful_factor,
    )

    attack_mult = max(s.injury_min_attack_mult, 1.0 - out_attack - dout_attack)
    defense_mult = min(s.injury_max_defense_mult, 1.0 + out_defense + dout_defense)
    return attack_mult, defense_mult
