"""Mapeo unificado de nombres de equipos entre fuentes.

Fuentes soportadas:
- Oloraculo (repo de referencia con bracket del WC 2026)
- martj42/international_results (CSV principal)
- StatsBomb (xG open data)
- football-data.co.uk (CSVs historicos de ligas, no usado en prod)

Reglas:
- La clave es el nombre alternativo, el valor es el nombre canónico (martj42).
- El nombre canónico es el que aparece en el CSV de martj42, que es nuestro source-of-truth.
- Si dos fuentes usan el mismo nombre, no se lista (ej. "Brazil" esta en ambos).
"""
from __future__ import annotations

from typing import Final

# Mapeo alternativo -> canonico (martj42)
# Ambos sentidos (OLO->MARTJ y MARTJ->OLO) se derivan de este dict.
TEAM_NAME_ALIASES: Final[dict[str, str]] = {
    # Oloraculo -> martj42
    "USA": "United States",
    "South Korea": "South Korea",  # identity
    "Czechia": "Czech Republic",
    "Ivory Coast": "Ivory Coast",  # identity
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",  # martj42 usa "and" (no ampersand)
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Cape Verde": "Cape Verde",  # identity
    "Curacao": "Curaçao",  # con tilde
    # StatsBomb / football-data
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
}


def normalize(name: str) -> str:
    """Devuelve el nombre canonico (martj42) para el nombre dado.

    Si no esta en el mapeo, devuelve el nombre sin cambios.
    """
    return TEAM_NAME_ALIASES.get(name, name)


# Alias para compatibilidad con codigo que importa normalize_team_name
normalize_team_name = normalize


def denormalize(canonical: str) -> str:
    """Devuelve el primer nombre alternativo conocido para el canonico.

    Si no hay alias, devuelve el canonico sin cambios.
    Util para mostrar el nombre "Oloraculo" desde un nombre martj42.
    """
    for alias, canon in TEAM_NAME_ALIASES.items():
        if canon == canonical and alias != canonical:
            return alias
    return canonical


# Mapeo directo Oloraculo -> martj42 (todos los aliases)
# Mantener por compatibilidad con codigo existente que lo importa.
OLO_TO_MARTJ: Final[dict[str, str]] = dict(TEAM_NAME_ALIASES)
