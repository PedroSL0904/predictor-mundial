"""CLI: generar predicciones R32 del WC 2026.

Uso:
    python -m src.cli.wc2026_r32
"""
from __future__ import annotations

import time

import pandas as pd

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.data.injuries import load_injuries
from src.data.team_names import OLO_TO_MARTJ
from src.data.wc2026_fixture import generate_group_fixtures
from src.features.strengths_cache import StrengthsCache
from src.simulation.r32_predictions import (
    build_r32_matches,
    format_r32_table,
    format_standings_table,
)
from src.simulation.wc2026_simulate import TournamentSimulator


def main() -> None:
    from src.paths import (
        ELO_TIMELINE_JSON,
        MARTJ_CSV,
        R32_PREDICTIONS_CSV,
        README_R32,
        TEMPERATURE_CALIBRATOR,
    )
    csv_path = MARTJ_CSV
    cache_path = ELO_TIMELINE_JSON
    cal_path = TEMPERATURE_CALIBRATOR

    print("Cargando datos...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    calibrator = None
    if cal_path.exists():
        from src.models.calibration import TemperatureScaler
        calibrator = TemperatureScaler.load(cal_path)
        print(f"  Calibrador: T={calibrator.T_:.3f}")

    # as_of = un dia despues del ultimo partido FT
    fixtures_full = generate_group_fixtures()
    last_played = pd.to_datetime(
        fixtures_full[fixtures_full["played"]]["date"]
    ).max()
    as_of = (last_played + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  as_of = {as_of}")

    sim = TournamentSimulator(
        df, timeline, StrengthsCache(df, timeline),
        as_of=as_of, calibrator=calibrator,
        injuries=load_injuries(),
    )
    print("  Cache de predicciones listo")

    def predict_fn(home_olo: str, away_olo: str) -> dict:
        mp = sim.predict(home_olo, away_olo)
        return {
            "p_h": mp.p_home,
            "p_d": mp.p_draw,
            "p_a": mp.p_away,
            "most_likely": mp.most_likely,
        }

    # Construir partidos R32
    print("\nCalculando bracket R32...", flush=True)
    matches, standings, top_8_thirds = build_r32_matches(
        fixtures_full, OLO_TO_MARTJ, predict_fn,
    )

    # Imprimir resultados
    print(f"\n{'='*80}")
    print(f"R32 DEL WC 2026 - {len(matches)} partidos")
    print(f"{'='*80}\n")
    print(format_r32_table(matches))

    # Guardar CSV
    rows = []
    for m in matches:
        rows.append({
            "tie_id": m.tie_id,
            "home": m.home_team,
            "home_source": m.home_source,
            "p_home": m.p_home,
            "p_draw": m.p_draw,
            "p_away": m.p_away,
            "away": m.away_team,
            "away_source": m.away_source,
            "most_likely": f"{m.most_likely[0]}-{m.most_likely[1]}",
        })
    df_out = pd.DataFrame(rows)
    out_csv = R32_PREDICTIONS_CSV
    df_out.to_csv(out_csv, index=False)
    print(f"\nGuardado en {out_csv}")

    # Guardar grupo stage summary
    lines = [
        "# WC 2026 - Resumen de fase de grupos y R32",
        "",
        f"_Generado {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}._",
        "",
        "## Standings de grupos",
        "",
        format_standings_table(standings),
        "",
        "## 8 mejores terceros (califican a R32)",
        "",
        "| Grupo | Equipo | PTS | GD | GF |",
        "|---|---|---:|---:|---:|",
    ]
    for g, team, pts, gd, gf in top_8_thirds:
        gd_str = f"+{gd}" if gd > 0 else str(gd)
        lines.append(f"| {g} | {team} | {pts} | {gd_str} | {gf} |")
    lines.extend([
        "",
        "## Predicciones R32",
        "",
        format_r32_table(matches),
        "",
    ])
    out_md = README_R32
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown guardado en {out_md}")


if __name__ == "__main__":
    main()
