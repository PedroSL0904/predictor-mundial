"""Bracket del WC 2026: cruces de R32, R16, QF, SF y Final.

Formato: 12 grupos de 4 -> top 2 + 8 mejores terceros -> R32 -> ... -> Final.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SlotKind(StrEnum):
    GROUP_WINNER = "winner"
    GROUP_RUNNER_UP = "runner_up"
    GROUP_THIRD = "third"
    WINNER_OF = "winner_of"


@dataclass
class BracketSlot:
    kind: SlotKind
    group: str | None = None
    third_options: list[str] = field(default_factory=list)
    tie_id: int | None = None


@dataclass
class BracketTie:
    id: int
    home: BracketSlot
    away: BracketSlot


def _w(g: str) -> BracketSlot:
    return BracketSlot(kind=SlotKind.GROUP_WINNER, group=g)


def _r(g: str) -> BracketSlot:
    return BracketSlot(kind=SlotKind.GROUP_RUNNER_UP, group=g)


def _t(*gs: str) -> BracketSlot:
    return BracketSlot(kind=SlotKind.GROUP_THIRD, third_options=list(gs))


def _wo(tid: int) -> BracketSlot:
    return BracketSlot(kind=SlotKind.WINNER_OF, tie_id=tid)


# R32 (16 partidos, IDs 73-88)
ROUND_OF_32 = [
    BracketTie(73, _r("A"), _r("B")),
    BracketTie(74, _w("E"), _t("A", "B", "C", "D", "F")),
    BracketTie(75, _w("F"), _r("C")),
    BracketTie(76, _w("C"), _r("F")),
    BracketTie(77, _w("I"), _t("C", "D", "F", "G", "H")),
    BracketTie(78, _r("E"), _r("I")),
    BracketTie(79, _w("A"), _t("C", "E", "F", "H", "I")),
    BracketTie(80, _w("L"), _t("E", "H", "I", "J", "K")),
    BracketTie(81, _w("D"), _t("B", "E", "F", "I", "J")),
    BracketTie(82, _w("G"), _t("A", "E", "H", "I", "J")),
    BracketTie(83, _r("K"), _r("L")),
    BracketTie(84, _w("H"), _r("J")),
    BracketTie(85, _w("B"), _t("E", "F", "G", "I", "J")),
    BracketTie(86, _w("J"), _r("H")),
    BracketTie(87, _w("K"), _t("D", "E", "I", "J", "L")),
    BracketTie(88, _r("D"), _r("G")),
]

# R16 (8 partidos, IDs 89-96)
ROUND_OF_16 = [
    BracketTie(89, _wo(74), _wo(77)),
    BracketTie(90, _wo(73), _wo(75)),
    BracketTie(91, _wo(76), _wo(78)),
    BracketTie(92, _wo(79), _wo(80)),
    BracketTie(93, _wo(83), _wo(84)),
    BracketTie(94, _wo(81), _wo(82)),
    BracketTie(95, _wo(86), _wo(88)),
    BracketTie(96, _wo(85), _wo(87)),
]

# QF (4 partidos, IDs 97-100)
QUARTER_FINALS = [
    BracketTie(97, _wo(89), _wo(90)),
    BracketTie(98, _wo(93), _wo(94)),
    BracketTie(99, _wo(91), _wo(92)),
    BracketTie(100, _wo(95), _wo(96)),
]

# SF (2 partidos, IDs 101-102)
SEMI_FINALS = [
    BracketTie(101, _wo(97), _wo(98)),
    BracketTie(102, _wo(99), _wo(100)),
]

# Final (ID 104, ID 103 = 3rd place)
FINAL = BracketTie(104, _wo(101), _wo(102))


# Tabla oficial FIFA WC 2026: asignacion de los 8 mejores terceros a los slots R32.
# Basada en el bracket publicado por FIFA. El rank 1 = mejor tercero, rank 8 = peor.
# El matchup es: 1st del grupo X vs 3rd del grupo Y segun esta tabla fija.
#
#   tie_id -> grupo del 3ro (letra A-L)
#   Ej: 74 -> "D" significa que en el partido #74, el 3ro del grupo D juega
#       contra el winner del grupo E (1E vs 3D = Germany vs Paraguay).
#
# Validada con el bracket oficial (imagen del usuario, 28-jun-2026).
FIFA_THIRD_ASSIGNMENT: dict[int, str] = {
    74: "D",   # 1E (GER) vs 3D (PAR)
    77: "F",   # 1I (FRA) vs 3F (SWE)
    79: "E",   # 1A (MEX) vs 3E (ECU)
    80: "K",   # 1L (ENG) vs 3K (COD)
    81: "B",   # 1D (USA) vs 3B (BIH)
    82: "I",   # 1G (BEL) vs 3I (SEN)
    85: "J",   # 1B (SUI) vs 3J (ALG)
    87: "L",   # 1K (COL) vs 3L (GHA)
}


def assign_third_place_slots(qualified_third_groups: list[str]) -> dict[int, str]:
    """Asigna los grupos terceros a los slots del R32 segun la tabla oficial FIFA.

    Args:
        qualified_third_groups: 8 grupos (letras A-L) que pasan como terceros.
            El orden NO importa: la tabla FIFA es fija independiente del rank.

    Returns:
        dict[tie_id, group]: tie_id del R32 -> grupo asignado.
        None si algun grupo de FIFA_THIRD_ASSIGNMENT no esta en qualified.
    """
    if len(qualified_third_groups) != 8:
        return None

    qualified = set(qualified_third_groups)
    result: dict[int, str] = {}
    for tie_id, group in FIFA_THIRD_ASSIGNMENT.items():
        if group not in qualified:
            return None  # El bracket FIFA no aplica a estos 8 terceros
        result[tie_id] = group
    return result


def assign_third_place_slots_backtrack(qualified_third_groups: list[str]) -> dict[int, str] | None:
    """DEPRECATED: usar assign_third_place_slots (tabla FIFA fija).

    Backtracking alfabetico que producia asignaciones NO canonicas.
    Mantenido por compatibilidad con tests legacy.
    """
    if len(qualified_third_groups) != 8:
        return None

    qualified = set(qualified_third_groups)
    third_slots = []
    for tie in ROUND_OF_32:
        if tie.home.kind == SlotKind.GROUP_THIRD:
            third_slots.append((tie.id, "home", tie.home.third_options))
        if tie.away.kind == SlotKind.GROUP_THIRD:
            third_slots.append((tie.id, "away", tie.away.third_options))

    third_slots.sort(key=lambda x: (len(x[2]), x[0], x[1]))

    assigned: dict[tuple[int, str], str] = {}
    used: set[str] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(third_slots):
            return True
        tie_id, side, options = third_slots[idx]
        for g in sorted(options):
            if g in qualified and g not in used:
                used.add(g)
                assigned[(tie_id, side)] = g
                if backtrack(idx + 1):
                    return True
                del assigned[(tie_id, side)]
                used.remove(g)
        return False

    if backtrack(0):
        result_back: dict[int, str] = {}
        for (tie_id, _side), g in assigned.items():
            result_back[tie_id] = g
        return result_back
    return None
