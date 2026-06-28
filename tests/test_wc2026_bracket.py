"""Tests del bracket WC 2026 y asignacion FIFA de terceros."""
from __future__ import annotations

from src.simulation.wc2026_bracket import (
    FIFA_THIRD_ASSIGNMENT,
    ROUND_OF_32,
    SlotKind,
    assign_third_place_slots,
    assign_third_place_slots_backtrack,
)

# 8 mejores terceros (orden basado en pts, GD, GF segun el WC 2026 actual)
QUALIFIED_THIRDS = ["K", "F", "E", "L", "B", "J", "D", "I"]


def test_bracket_has_16_r32_matches() -> None:
    assert len(ROUND_OF_32) == 16


def test_bracket_tie_ids_73_to_88() -> None:
    ids = sorted(t.id for t in ROUND_OF_32)
    assert ids == list(range(73, 89))


def test_bracket_has_exactly_8_third_slots() -> None:
    n_third = sum(
        1 for tie in ROUND_OF_32
        for slot in (tie.home, tie.away)
        if slot.kind == SlotKind.GROUP_THIRD
    )
    assert n_third == 8


def test_fifa_assignment_covers_all_third_slots() -> None:
    """Los tie_ids en FIFA_THIRD_ASSIGNMENT deben coincidir con los slots third."""
    third_tie_ids = {
        tie.id
        for tie in ROUND_OF_32
        if tie.home.kind == SlotKind.GROUP_THIRD
        or tie.away.kind == SlotKind.GROUP_THIRD
    }
    assert set(FIFA_THIRD_ASSIGNMENT.keys()) == third_tie_ids


def test_fifa_assignment_uses_distinct_groups() -> None:
    """Los 8 grupos asignados deben ser distintos."""
    assert len(set(FIFA_THIRD_ASSIGNMENT.values())) == 8


def test_assign_third_place_slots_returns_fifa_table() -> None:
    """Con 8 terceros calificados, devuelve la tabla FIFA exacta."""
    result = assign_third_place_slots(QUALIFIED_THIRDS)
    assert result == FIFA_THIRD_ASSIGNMENT


def test_assign_third_place_slots_wrong_count() -> None:
    assert assign_third_place_slots(["A", "B"]) is None
    assert assign_third_place_slots([]) is None


def test_assign_third_place_slots_missing_fifa_group() -> None:
    """Si falta un grupo que la FIFA necesita, devuelve None."""
    # Sacar K (rank 1) de qualified
    no_k = [g for g in QUALIFIED_THIRDS if g != "K"]
    assert assign_third_place_slots(no_k) is None


def test_assign_third_place_slots_with_extra_group() -> None:
    """Si hay 9 grupos en qualified (no deberia), igual devuelve FIFA table."""
    extras = QUALIFIED_THIRDS + ["X"]
    result = assign_third_place_slots(extras)
    # Como el check es len == 8, debe fallar
    assert result is None


def test_fifa_specific_matchups() -> None:
    """Verifica los matchups especificos que el usuario corrigio."""
    result = assign_third_place_slots(QUALIFIED_THIRDS)
    # P74: 1E (GER) vs 3D (PAR)
    assert result[74] == "D"
    # P77: 1I (FRA) vs 3F (SWE)
    assert result[77] == "F"
    # P79: 1A (MEX) vs 3E (ECU)
    assert result[79] == "E"
    # P80: 1L (ENG) vs 3K (COD)
    assert result[80] == "K"
    # P81: 1D (USA) vs 3B (BIH)
    assert result[81] == "B"
    # P82: 1G (BEL) vs 3I (SEN)
    assert result[82] == "I"
    # P85: 1B (SUI) vs 3J (ALG)
    assert result[85] == "J"
    # P87: 1K (COL) vs 3L (GHA)
    assert result[87] == "L"


def test_backtrack_deprecated_still_works() -> None:
    """El backtrack DEPRECATED sigue funcionando para backward compat."""
    result = assign_third_place_slots_backtrack(QUALIFIED_THIRDS)
    assert result is not None
    # El backtrack da una asignacion valida (no necesariamente FIFA)
    assert set(result.keys()) == set(FIFA_THIRD_ASSIGNMENT.keys())
    assert set(result.values()) == set(QUALIFIED_THIRDS)


def test_bracket_r16_uses_winner_of() -> None:
    """Los partidos R16 deben referenciar winner_of a partidos R32."""
    from src.simulation.wc2026_bracket import ROUND_OF_16
    for tie in ROUND_OF_16:
        for slot in (tie.home, tie.away):
            if slot.kind == SlotKind.WINNER_OF:
                assert 73 <= slot.tie_id <= 88  # apunta a R32
