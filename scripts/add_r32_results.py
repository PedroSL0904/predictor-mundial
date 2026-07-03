"""Agrega los resultados R32 del WC 2026 al CSV de martj42.

Los partidos se juegan entre 2026-07-01 y 2026-07-05.
"""
from __future__ import annotations

from pathlib import Path

CSV_PATH = Path("data/raw/martj42_results.csv")

# Resultados R32 confirmados (fecha, home, away, home_score, away_score, city)
# Teams en formato martj42 (no se traduce):
#   - USA = "United States"
#   - Bosnia & Herzegovina = "Bosnia and Herzegovina"
#   - Ivory Coast (no "Cote d'Ivoire")
#   - Cape Verde
R32_RESULTS = [
    # 2026-07-01: P73 RSA-CAN
    ("2026-07-01", "South Africa", "Canada", 0, 1, "Houston"),
    # 2026-07-02: P74 GER-PAR, P75 NED-MAR (pens 3-1, FT 1-1)
    ("2026-07-02", "Germany", "Paraguay", 3, 1, "Philadelphia"),
    ("2026-07-02", "Netherlands", "Morocco", 1, 1, "East Rutherford"),
    # 2026-07-03: P77 FRA-SWE, P78 CIV-NOR
    ("2026-07-03", "France", "Sweden", 3, 0, "Boston"),
    ("2026-07-03", "Ivory Coast", "Norway", 1, 2, "Miami Gardens"),
    # 2026-07-04: P79 MEX-ECU, P80 ENG-COD, P81 USA-BIH
    ("2026-07-04", "Mexico", "Ecuador", 2, 0, "Atlanta"),
    ("2026-07-04", "England", "DR Congo", 2, 1, "Kansas City"),
    ("2026-07-04", "United States", "Bosnia and Herzegovina", 2, 0, "Los Angeles"),
    # 2026-07-05: P82 BEL-SEN, P83 POR-CRO
    ("2026-07-05", "Belgium", "Senegal", 3, 2, "Dallas"),
    ("2026-07-05", "Portugal", "Croatia", 2, 1, "Seattle"),
    # 2026-07-06: P84 ESP-AUT, P85 SUI-ALG
    ("2026-07-06", "Spain", "Austria", 3, 0, "San Francisco"),
    ("2026-07-06", "Switzerland", "Algeria", 2, 0, "Denver"),
    # 2026-07-07: P76 BRA-JPN, P86 ARG-CPV
    ("2026-07-07", "Brazil", "Japan", 2, 1, "New York/New Jersey"),
    ("2026-07-07", "Argentina", "Cape Verde", 2, 0, "Chicago"),
]


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV no existe: {CSV_PATH}")

    existing = CSV_PATH.read_text(encoding="utf-8")
    lines = existing.splitlines()
    header = lines[0]
    body = lines[1:]

    # Detectar duplicados
    existing_keys = set()
    for line in body:
        parts = line.split(",")
        if len(parts) >= 5:
            key = (parts[0], parts[1], parts[2])
            existing_keys.add(key)

    new_lines = []
    added = 0
    skipped = 0
    for date, home, away, hs, ascore, city in R32_RESULTS:
        key = (date, home, away)
        if key in existing_keys:
            skipped += 1
            continue
        # formato: date,home,away,hg,ag,tournament,city,country,neutral
        new_lines.append(
            f"{date},{home},{away},{float(hs)},{float(ascore)},"
            f"FIFA World Cup,{city},United States,True"
        )
        added += 1

    with CSV_PATH.open("a", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    print(f"Agregados: {added} | Duplicados saltados: {skipped}")
    print(f"Total: {len(body) + added} matches en {CSV_PATH}")


if __name__ == "__main__":
    main()
