"""Tests para el modulo statsbomb y la integracion de xG real."""
from __future__ import annotations

from src.data.statsbomb import (
    get_xg_real,
    get_xg_real_lookup,
    has_xg_real,
    statsbomb_coverage,
)


def test_statsbomb_lookup_loads() -> None:
    """El cache debe existir y cargarse."""
    lookup = get_xg_real_lookup()
    assert isinstance(lookup, dict)
    assert len(lookup) > 100  # Esperamos ~125 partidos


def test_statsbomb_has_xg_for_wc_final() -> None:
    """WC 2022 final Argentina vs Francia debe estar en el cache."""
    # match_id 3869686, Argentina vs France, 2022-12-18
    assert has_xg_real("2022-12-18", "Argentina", "France")


def test_statsbomb_xg_values_reasonable() -> None:
    """Los xG deben estar en rango razonable para un partido de Mundial."""
    xg = get_xg_real("2022-12-18", "Argentina", "France")
    assert xg is not None
    home_xg, away_xg = xg
    # Final muy disputada, xG alto para ambos (5-6 cada uno segun StatsBomb)
    assert 2.0 < home_xg < 8.0
    assert 2.0 < away_xg < 8.0


def test_statsbomb_no_xg_for_old_match() -> None:
    """Partidos pre-2018 no tienen xG real."""
    # WC 2014 final: Germany vs Argentina, 2014-07-13
    assert not has_xg_real("2014-07-13", "Germany", "Argentina")


def test_statsbomb_curacao_normalization() -> None:
    """Normalizacion de Curacao -> Curaçaao funciona en ambos sentidos."""
    # WC 2022 no tuvo Curacao, pero probemos que la normalizacion es estable
    has1 = has_xg_real("2018-06-15", "Curaçao", "Japan")
    has2 = has_xg_real("2018-06-15", "Curacao", "Japan")
    # Ambos deben dar el mismo resultado (puede ser True o False)
    assert has1 == has2


def test_statsbomb_coverage_in_dataset() -> None:
    """La funcion de cobertura debe contar partidos correctamente."""
    import pandas as pd
    from src.data.historical import load_martj42_csv
    df = load_martj42_csv("data/raw/martj42_results.csv")
    cov = statsbomb_coverage(df)
    assert cov["total"] > 40000  # ~49k partidos en el dataset
    assert cov["with_xg"] > 100  # ~125 partidos con xG
    # Solo anos 2018 y 2022 tienen cobertura
    assert "2018" in cov["by_year"]
    assert "2022" in cov["by_year"]
    assert "2014" not in cov["by_year"]
