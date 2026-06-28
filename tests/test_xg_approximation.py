"""Tests para src/features/xg_approximation.py (unica implementacion)."""
import numpy as np

from src.features.xg_approximation import approx_xg, approx_xg_from_elo


def test_neutral_elo_returns_league_mean():
    """Elo=1500 vs Elo=1500 → xG cercano al league_mean (~1.30)."""
    assert abs(approx_xg_from_elo(1500, 1500) - 1.30) < 0.01


def test_stronger_attacker_higher_xg():
    """Atacante con Elo mayor que defensor → xG > league_mean."""
    assert approx_xg_from_elo(1700, 1500) > 1.30
    assert approx_xg_from_elo(1900, 1500) > approx_xg_from_elo(1700, 1500)


def test_weaker_attacker_lower_xg():
    """Atacante con Elo menor que defensor → xG < league_mean."""
    assert approx_xg_from_elo(1300, 1500) < 1.30
    assert approx_xg_from_elo(1100, 1500) < approx_xg_from_elo(1300, 1500)


def test_symmetry():
    """aprox_xg(a, b) > league_mean  ⇔  aprox_xg(b, a) < league_mean."""
    base = approx_xg_from_elo(1500, 1500)
    high = approx_xg_from_elo(1700, 1500)
    low = approx_xg_from_elo(1500, 1700)
    assert abs((high - base) - (base - low)) < 0.01  # simetria


def test_vectorized_matches_scalar():
    """Vectorizado y escalar deben dar el mismo resultado."""
    pairs = [(1500, 1500), (1700, 1500), (1300, 1500), (1800, 1600)]
    scalars = [approx_xg_from_elo(a, d) for a, d in pairs]
    arrs_a = np.array([p[0] for p in pairs])
    arrs_d = np.array([p[1] for p in pairs])
    vectorized = approx_xg(arrs_a, arrs_d)
    for s, v in zip(scalars, vectorized, strict=False):
        assert abs(s - float(v)) < 1e-10


def test_polymorphic_approx_xg_from_elo():
    """approx_xg_from_elo acepta escalares y arrays."""
    # Escalar
    val = approx_xg_from_elo(1500, 1500)
    assert isinstance(val, float)

    # Array
    arr = approx_xg_from_elo(np.array([1500, 1700]), np.array([1500, 1500]))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)
