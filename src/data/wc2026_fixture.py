"""Fixture del WC 2026: 12 grupos, 72 partidos de fase de grupos.

Extraido del README de Oloraculo (que tiene los 72 partidos con
predicciones y resultados parciales). Los grupos se infieren del README.
"""
from __future__ import annotations

import pandas as pd

# 12 grupos, cada uno con 4 equipos y 6 partidos
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


def generate_group_fixtures() -> pd.DataFrame:
    """Genera los 72 partidos de grupo con fechas estimadas.

    Fechas: el WC 2026 comenzo el 11 de junio. Partidos de grupo son
    del 11 al 27 de junio. Tomamos las fechas del CSV cuando estan
    disponibles, y estimamos las faltantes en orden.
    """
    import pandas as pd
    from pathlib import Path
    csv_path = Path(r"C:\dev\predictor-mundial\data\raw\martj42_results.csv")
    df = pd.read_csv(csv_path)
    wc = df[(df["date"] >= "2026-06-01") & (df["tournament"] == "FIFA World Cup")]
    wc = wc[wc["home_score"].notna() & wc["away_score"].notna()]

    martj_to_olo = {
        "South Korea": "South Korea",
        "South Africa": "South Africa",
        "Czech Republic": "Czechia",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Cape Verde": "Cape Verde",
        "Saudi Arabia": "Saudi Arabia",
        "DR Congo": "Congo DR",
        "Ivory Coast": "Ivory Coast",
        "United States": "United States",
        "Scotland": "Scotland",
        "England": "England",
        "Curaçao": "Curacao",
    }
    olo_to_martj = {v: k for k, v in martj_to_olo.items()}

    # Fechas estimadas para partidos pendientes (WC 2026, fase de grupos)
    # Matchday 1: 11-13 jun. Matchday 2: 17-19 jun. Matchday 3: 23-27 jun.
    # Distribuir los 12 grupos entre los dias disponibles
    ESTIMATED_DATES = {
        0: ["2026-06-11", "2026-06-12"],  # matchday 1
        2: ["2026-06-17", "2026-06-18", "2026-06-19"],  # matchday 2
        4: ["2026-06-23", "2026-06-25", "2026-06-27"],  # matchday 3
    }

    fixtures = []
    group_idx = 0
    for group, teams in GROUPS.items():
        # 6 partidos: 3 fechas (matchdays)
        pairs = [
            (0, 1), (2, 3),  # matchday 1 (par 0,1)
            (0, 2), (3, 1),  # matchday 2 (par 2,3)
            (0, 3), (1, 2),  # matchday 3 (par 4,5)
        ]
        for pair_idx, (h_idx, a_idx) in enumerate(pairs):
            home_olo = teams[h_idx]
            away_olo = teams[a_idx]
            home_martj = olo_to_martj.get(home_olo, home_olo)
            away_martj = olo_to_martj.get(away_olo, away_olo)

            # Buscar fecha en CSV (partido ya jugado)
            match_row = wc[
                (wc["home_team"] == home_martj) & (wc["away_team"] == away_martj)
            ]
            if not match_row.empty:
                date = str(match_row["date"].iloc[0])[:10]
                home_score = int(match_row["home_score"].iloc[0])
                away_score = int(match_row["away_score"].iloc[0])
                played = True
            else:
                # Estimar fecha segun matchday y grupo
                md = pair_idx // 2  # 0, 1, 2
                dates_for_md = ESTIMATED_DATES.get(md * 2, ["2026-06-20"])
                # Distribuir grupos entre fechas
                date_idx = group_idx % len(dates_for_md)
                date = dates_for_md[date_idx]
                home_score = None
                away_score = None
                played = False

            fixtures.append({
                "group": group,
                "home": home_olo,
                "away": away_olo,
                "home_martj": home_martj,
                "away_martj": away_martj,
                "date": date,
                "home_score": home_score,
                "away_score": away_score,
                "played": played,
            })
        group_idx += 1

    return pd.DataFrame(fixtures)


if __name__ == "__main__":
    fx = generate_group_fixtures()
    print(f"Total fixtures: {len(fx)}")
    print(f"Played: {fx['played'].sum()}")
    print(f"Pending: {(~fx['played']).sum()}")
    print()
    print("Ejemplo grupo A:")
    print(fx[fx.group == "A"][["date", "home", "away", "home_score", "away_score", "played"]].to_string(index=False))
    print()
    print("Fechas pendientes (estimadas):")
    pending = fx[~fx.played]
    print(pending[["group", "home", "away"]].head(10).to_string(index=False))
