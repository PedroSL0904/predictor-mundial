"""Actualiza el CSV con los resultados R32 y R16 reales del WC 2026.

Estrategia:
1. Borra las filas R32 existentes (fechas 2026-07-01 a 2026-07-07)
2. Re-inserta con fechas correctas y scores actualizados
3. Agrega P88 (AUS-EGY) que faltaba
4. Agrega 4 R16 (W89, W90, W91, W92)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/raw/martj42_results.csv")

# R32 confirmados (fecha, home, away, hs, as, city, tie_id)
# tie_id agregado para tracking (no es parte del CSV)
R32_FINAL = [
    # Wed 2026-07-01
    ("2026-07-01", "Argentina", "Cape Verde", 3, 2, "Chicago", 86),  # AET (2-2 90min)
    # Thu 2026-07-02
    ("2026-07-02", "Portugal", "Croatia", 2, 1, "Seattle", 83),
    ("2026-07-02", "Spain", "Austria", 3, 0, "San Francisco", 84),
    ("2026-07-02", "Switzerland", "Algeria", 2, 0, "Denver", 85),
    # Fri 2026-07-03
    ("2026-07-03", "Australia", "Egypt", 1, 1, "Houston", 88),  # Egypt won 4-2 pens
    ("2026-07-03", "Colombia", "Ghana", 1, 0, "Miami Gardens", 87),
    # (los demas R32 ya estaban correctos)
    ("2026-07-01", "South Africa", "Canada", 0, 1, "Houston", 73),
    ("2026-07-01", "Germany", "Paraguay", 3, 1, "Philadelphia", 74),
    ("2026-07-02", "Netherlands", "Morocco", 1, 1, "East Rutherford", 75),  # MAR pens
    ("2026-07-02", "France", "Sweden", 3, 0, "Boston", 77),
    ("2026-07-02", "Ivory Coast", "Norway", 1, 2, "Miami Gardens", 78),
    ("2026-07-03", "Mexico", "Ecuador", 2, 0, "Atlanta", 79),
    ("2026-07-03", "England", "DR Congo", 2, 1, "Kansas City", 80),
    ("2026-07-03", "United States", "Bosnia and Herzegovina", 2, 0, "Los Angeles", 81),
    ("2026-07-03", "Belgium", "Senegal", 3, 2, "Dallas", 82),
    ("2026-07-03", "Brazil", "Japan", 2, 1, "New York/New Jersey", 76),
]

# R16 confirmados (al 5-jul-2026)
R16_FINAL = [
    # Sat 2026-07-04
    ("2026-07-04", "Canada", "Morocco", 0, 3, "Houston", 90),
    ("2026-07-04", "Paraguay", "France", 0, 1, "Philadelphia", 89),
    # Sun 2026-07-05
    ("2026-07-05", "Brazil", "Norway", 1, 2, "Boston", 91),  # Norway upset
    ("2026-07-05", "Mexico", "England", 2, 3, "Atlanta", 92),  # England upset
]


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV no existe: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"CSV inicial: {len(df)} filas")

    # Detectar y borrar filas R32 existentes (fechas 2026-07-01 a 2026-07-07, WC 2026)
    mask_r32 = (
        (df["date"] >= "2026-07-01") & (df["date"] <= "2026-07-07")
        & (df["tournament"] == "FIFA World Cup")
    )
    n_r32_exist = mask_r32.sum()
    print(f"R32 existentes a borrar: {n_r32_exist}")
    df = df[~mask_r32].reset_index(drop=True)

    # Re-agregar R32 + R16
    new_rows = []
    for date, home, away, hs, ag, city, tie_id in R32_FINAL + R16_FINAL:
        new_rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": float(hs),
            "away_score": float(ag),
            "tournament": "FIFA World Cup",
            "city": city,
            "country": "United States",
            "neutral": True,
        })

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"CSV final: {len(df)} filas")
    print(f"  - Agregados: {len(R32_FINAL)} R32 (reemplazo) + {len(R16_FINAL)} R16 (nuevos)")

    # Verificar
    wc = df[(df["date"] >= "2026-06-01") & (df["tournament"] == "FIFA World Cup")].copy()
    wc = wc.dropna(subset=["home_score", "away_score"])
    print(f"  - Total WC 2026 played: {len(wc)}")
    by_date = wc.groupby(wc["date"].dt.strftime("%Y-%m-%d")).size()
    print(f"  - Por dia:")
    for d, n in by_date.items():
        print(f"      {d}: {n}")


if __name__ == "__main__":
    main()
