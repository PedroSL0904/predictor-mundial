"""Aproximacion de xG (expected goals) basada en diferencia de Elo.

Consolidacion: existia duplicado en 3 archivos:
- src/features/strengths.py:189 (escalar)
- src/features/strengths_cache.py:24 (vectorizado)
- src/evaluation/grid_search_fast.py:128 (escalar duplicado)

Aqui consolidamos en 2 funciones: una vectorizada (numpy) y una escalar
(usa la vectorizada internamente para mantener consistencia).
"""
from __future__ import annotations

import numpy as np

from src.config import get_settings


def approx_xg(elo_attacker, elo_defender):
    """Aproxima expected goals del atacante vs defensor.

    Acepta tanto escalares (float/int) como arrays numpy. Usada tanto
    en el path vectorizado (StrengthsCache) como en llamadas individuales
    (strengths.py, grid_search_fast.py).

    Args:
        elo_attacker: Elo del atacante (escalar o array).
        elo_defender: Elo del defensor (escalar o array).

    Returns:
        xG esperado (goles/partido). Mismo tipo que la entrada.
    """
    s = get_settings()
    base = s.league_mean
    return base * (1 + 0.30 * np.tanh((elo_attacker - elo_defender) / s.elo_divisor))


def approx_xg_from_elo(elo_attacker, elo_defender):
    """Aproxima xG para un par (atacante, defensor). Polimorfico.

    Acepta float, int, np.ndarray (cualquier dimension). Retorna
    el mismo tipo que la entrada.
    """
    result = approx_xg(elo_attacker, elo_defender)
    # Si la entrada era array, devolver array
    if isinstance(elo_attacker, np.ndarray):
        return result
    # Si era escalar, devolver float
    return float(result)
