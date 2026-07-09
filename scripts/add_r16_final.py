"""Agrega los 4 R16 faltantes al CSV (W93, W94, W95, W96).

W95 Switzerland 0-0 Colombia, Switzerland gano 4-3 en penalties.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/raw/martj42_results.csv")

# R16 W93-W96
R16_FINAL = [
    # Mon 2026-07-06
    ("2026-07-06", "Portugal", "Spain", 0, 1, "Seattle", 93),
    ("2026-07-06", "United States", "Belgium", 1, 4, "Houston", 94),
    ("2026-07-06", "Switzerland", "Colombia", 0, 0, "Miami Gardens", 95),  # SUI 4-3 pens
    ("2026-07-06", "Argentina", "Egypt", 3, 2, "East Rutherford", 96),
]


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV no existe: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"CSV inicial: {len(df)} filas")

    # Verificar duplicados
    existing_keys = set()
    for _, row in df.iterrows():
        existing_keys.add((str(row["date"])[:10], row["home_team"], row["away_team"]))

    new_rows = []
    for date, home, away, hs, ag, city, tie_id in R16_FINAL:
        key = (date, home, away)
        if key in existing_keys:
            print(f"  Skipping duplicate: {key}")
            continue
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
    print(f"  - Agregados: {len(new_rows)} R16")

    # Verificar
    wc = df[(df["date"] >= "2026-06-01") & (df["tournament"] == "FIFA World Cup")].copy()
    wc = wc.dropna(subset=["home_score", "away_score"])
    print(f"  - Total WC 2026 played: {len(wc)}")


if __name__ == "__main__":
    main()
