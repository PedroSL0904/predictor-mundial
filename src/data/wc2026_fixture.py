"""Fixture del WC 2026: 12 grupos, 72 partidos de fase de grupos.

Lee los partidos directamente del CSV (que tiene los 72 partidos con
fechas reales). Los grupos se mantienen hardcoded ya que no estan
en el CSV. Para partidos pendientes (sin resultado), se infiere
el grupo y se usa la fecha del CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.team_names import OLO_TO_MARTJ

# 12 grupos, cada uno con 4 equipos
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Qatar", "Bosnia and Herzegovina", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Saudi Arabia", "Cape Verde", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def generate_group_fixtures(csv_path: Path | None = None) -> pd.DataFrame:
    """Genera los 72 partidos de grupo desde el CSV de martj42.

    El CSV tiene los 72 partidos del WC 2026 con fechas reales y
    resultados parciales. Mapeamos los equipos a sus grupos y nombres
    Oloraculo.
    """
    if csv_path is None:
        from src.paths import MARTJ_CSV
        csv_path = MARTJ_CSV

    df = pd.read_csv(csv_path)
    wc = df[(df["date"] >= "2026-06-01") & (df["tournament"] == "FIFA World Cup")].copy()
    wc = wc.sort_values("date").reset_index(drop=True)

    # Crear reverse map: martj_name -> olo_name
    martj_to_olo = {v: k for k, v in OLO_TO_MARTJ.items()}
    # Y group lookup
    team_to_group = {}
    team_to_olo = {}
    for group, teams in GROUPS.items():
        for t in teams:
            team_to_group[t] = group
            team_to_olo[t] = t

    fixtures = []
    for _, row in wc.iterrows():
        home_martj = row["home_team"]
        away_martj = row["away_team"]
        # Mapear a olo
        home_olo = martj_to_olo.get(home_martj, home_martj)
        away_olo = martj_to_olo.get(away_martj, away_martj)
        # Buscar grupo
        group = team_to_group.get(home_olo) or team_to_group.get(home_martj)
        if not group:
            # No es un partido de grupo (puede ser eliminatoria), skip
            continue

        date = str(row["date"])[:10]
        played = pd.notna(row["home_score"]) and pd.notna(row["away_score"])
        fixtures.append({
            "group": group,
            "home": home_olo,
            "away": away_olo,
            "home_martj": home_martj,
            "away_martj": away_martj,
            "date": date,
            "home_score": int(row["home_score"]) if played else None,
            "away_score": int(row["away_score"]) if played else None,
            "played": played,
        })

    return pd.DataFrame(fixtures)


if __name__ == "__main__":
    fx = generate_group_fixtures()
    print(f"Total fixtures: {len(fx)}")
    print(f"Played: {fx['played'].sum()}")
    print(f"Pending: {(~fx['played']).sum()}")
    print()
    print("Ejemplo grupo A:")
    print(fx[fx.group == "A"][["date", "home", "away", "home_score", "away_score", "played"]].to_string(index=False))
