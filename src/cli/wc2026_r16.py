"""Genera predicciones de R16, QF, SF y Final del WC 2026.

Toma en cuenta los resultados R32 ya jugados para que el modelo
use el contexto actualizado al predecir las siguientes rondas.
"""
from __future__ import annotations

import pandas as pd

from src.cli.wc2026_readme import (
    _compute_as_of_from_csv,
    _load_data,
    predict_match,
)
from src.data.injuries import load_injuries
from src.logging_config import get_logger, setup_logging
from src.models.calibration import TemperatureScaler
from src.paths import (
    R16_PREDICTIONS_CSV,
    README_R16,
    TEMPERATURE_CALIBRATOR,
)

logger = get_logger(__name__)

# Resultados R32 (winner de cada P-id)
# Clave: P-id (73-88), Valor: nombre martj del ganador
R32_WINNERS = {
    73: "Canada",         # 1-0 vs South Africa
    74: "Germany",        # 3-1 vs Paraguay
    75: "Morocco",        # pens (1-1) vs Netherlands
    76: "Brazil",         # 2-1 vs Japan
    77: "France",         # 3-0 vs Sweden
    78: "Norway",         # 2-1 vs Ivory Coast
    79: "Mexico",         # 2-0 vs Ecuador
    80: "England",        # 2-1 vs DR Congo
    81: "United States",  # 2-0 vs Bosnia and Herzegovina
    82: "Belgium",        # 3-2 vs Senegal
    83: "Portugal",       # 2-1 vs Croatia
    84: "Spain",          # 3-0 vs Austria
    85: "Switzerland",    # 2-0 vs Algeria
    86: "Argentina",      # 3-2 AET vs Cape Verde
    87: "Colombia",       # 1-0 vs Ghana
    88: "Egypt",          # 1-1 (4-2 pens) vs Australia
}

# R16 ya jugados (al 5-jul-2026) - W89..W92
R16_KNOWN_WINNERS = {
    89: "France",    # 1-0 vs Paraguay
    90: "Morocco",   # 3-0 vs Canada
    91: "Norway",    # 2-1 vs Brazil (UPSET)
    92: "England",   # 3-2 vs Mexico (UPSET)
}

# R16 matchups (W-id): (winner de P-X, winner de P-Y)
# Per la imagen del bracket: W95 = SUI vs (winner P87), W96 = ARG vs (winner P88)
# (El codigo de wc2026_bracket.py tiene los IDs 95/96 swapped pero es
# indistinto para el QF P100, asi que usamos el orden del bracket oficial.)
R16_BRACKET = {
    89: (74, 77),  # GER vs FRA
    90: (73, 75),  # CAN vs MAR
    91: (76, 78),  # BRA vs NOR
    92: (79, 80),  # MEX vs ENG
    93: (83, 84),  # POR vs ESP
    94: (81, 82),  # USA vs BEL
    95: (85, 87),  # SUI vs (winner P87)
    96: (86, 88),  # ARG vs (winner P88)
}

# QF: (winner de W-X, winner de W-Y)
QF_BRACKET = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

# SF
SF_BRACKET = {
    101: (97, 98),
    102: (99, 100),
}

# Final
FINAL_BRACKET = (101, 102)


def main() -> None:
    setup_logging()
    logger.info("=" * 80)
    logger.info("PREDICCIONES R16, QF, SF, FINAL - WC 2026")
    logger.info("=" * 80)

    logger.info("Cargando datos...")
    df, timeline, cache = _load_data()
    calibrator = TemperatureScaler.load(TEMPERATURE_CALIBRATOR)
    logger.info(f"  Calibrador: T={calibrator.T_:.3f}")

    as_of = _compute_as_of_from_csv(df)
    logger.info(f"  as_of (post-R32) = {as_of}")

    injuries = load_injuries()

    # ---------- PREDECIR P87 y P88 (TBD R32) ----------
    # NOTA: P87 y P88 ya estan jugados (Colombia, Egypt). P87_pred y p88_pred
    # son solo para mostrar la prediccion original del modelo. Los winners
    # reales ya estan hardcoded en R32_WINNERS.
    logger.info("")
    logger.info("PREDICCION R32 (mostrar prediccion modelo vs realidad)")
    logger.info("-" * 80)
    p87_pred = predict_match(
        df, timeline, cache,
        "Colombia", "Ghana", "2026-07-07",
        as_of=as_of, calibrator=calibrator, injuries=injuries,
    )
    p88_pred = predict_match(
        df, timeline, cache,
        "Australia", "Egypt", "2026-07-07",
        as_of=as_of, calibrator=calibrator, injuries=injuries,
    )
    logger.info(f"P87 Colombia vs Ghana: H={p87_pred['p_h']:.0%} D={p87_pred['p_d']:.0%} A={p87_pred['p_a']:.0%} -> Real: Colombia")
    logger.info(f"P88 Australia vs Egypt: H={p88_pred['p_h']:.0%} D={p88_pred['p_d']:.0%} A={p88_pred['p_a']:.0%} -> Real: Egypt (pens)")

    # ---------- PREDECIR R16 ----------
    logger.info("")
    logger.info("PREDICCION R16 (OCTAVOS DE FINAL)")
    logger.info("=" * 80)
    r16_predictions = {}
    r16_winners = {}
    for w_id, (p_a, p_b) in R16_BRACKET.items():
        home = R32_WINNERS[p_a]
        away = R32_WINNERS[p_b]
        match_date = f"2026-07-{4 + (w_id - 89) // 2:02d}"  # 04, 05, 06, 07

        # Si el partido ya se jugo, usar winner conocido
        if w_id in R16_KNOWN_WINNERS:
            winner = R16_KNOWN_WINNERS[w_id]
            # Igual predecimos para mostrar la prob del modelo vs realidad
            pred = predict_match(
                df, timeline, cache,
                home, away, match_date,
                as_of=as_of, calibrator=calibrator, injuries=injuries,
            )
            r16_predictions[w_id] = {"home": home, "away": away, **pred}
            r16_winners[w_id] = winner
            actual_pick = home if pred["p_h"] >= max(pred["p_d"], pred["p_a"]) else away
            mark = "OK" if actual_pick == winner else "X"
            logger.info(
                f"W{w_id} {home} vs {away}: "
                f"H={pred['p_h']:.0%} D={pred['p_d']:.0%} A={pred['p_a']:.0%} | "
                f"**YA JUGADO** Winner real: {winner} (modelo pick: {actual_pick} {mark})"
            )
            continue

        pred = predict_match(
            df, timeline, cache,
            home, away, match_date,
            as_of=as_of, calibrator=calibrator, injuries=injuries,
        )
        r16_predictions[w_id] = {"home": home, "away": away, **pred}

        winner = home if pred["p_h"] >= max(pred["p_d"], pred["p_a"]) else away
        r16_winners[w_id] = winner

        logger.info(
            f"W{w_id} {home} vs {away}: "
            f"H={pred['p_h']:.0%} D={pred['p_d']:.0%} A={pred['p_a']:.0%} | "
            f"Winner: {winner}"
        )

    # ---------- PREDECIR QF ----------
    logger.info("")
    logger.info("PREDICCION QF (CUARTOS DE FINAL)")
    logger.info("=" * 80)
    qf_predictions = {}
    qf_winners = {}
    qf_dates = {97: "2026-07-14", 98: "2026-07-15", 99: "2026-07-14", 100: "2026-07-15"}
    for qf_id, (w_a, w_b) in QF_BRACKET.items():
        home = r16_winners[w_a]
        away = r16_winners[w_b]
        match_date = qf_dates[qf_id]

        pred = predict_match(
            df, timeline, cache,
            home, away, match_date,
            as_of=as_of, calibrator=calibrator, injuries=injuries,
        )
        qf_predictions[qf_id] = {"home": home, "away": away, **pred}
        winner = home if pred["p_h"] >= pred["p_a"] else away  # No draw in QF
        qf_winners[qf_id] = winner

        logger.info(
            f"QF{qf_id} {home} vs {away}: "
            f"H={pred['p_h']:.0%} D={pred['p_d']:.0%} A={pred['p_a']:.0%} | "
            f"Winner: {winner}"
        )

    # ---------- PREDECIR SF ----------
    logger.info("")
    logger.info("PREDICCION SF (SEMIFINALES)")
    logger.info("=" * 80)
    sf_predictions = {}
    sf_winners = {}
    sf_dates = {101: "2026-07-18", 102: "2026-07-19"}
    for sf_id, (qf_a, qf_b) in SF_BRACKET.items():
        home = qf_winners[qf_a]
        away = qf_winners[qf_b]
        match_date = sf_dates[sf_id]

        pred = predict_match(
            df, timeline, cache,
            home, away, match_date,
            as_of=as_of, calibrator=calibrator, injuries=injuries,
        )
        sf_predictions[sf_id] = {"home": home, "away": away, **pred}
        winner = home if pred["p_h"] >= pred["p_a"] else away
        sf_winners[sf_id] = winner

        logger.info(
            f"SF{sf_id} {home} vs {away}: "
            f"H={pred['p_h']:.0%} D={pred['p_d']:.0%} A={pred['p_a']:.0%} | "
            f"Winner: {winner}"
        )

    # ---------- PREDECIR FINAL ----------
    logger.info("")
    logger.info("PREDICCION FINAL")
    logger.info("=" * 80)
    home = sf_winners[101]
    away = sf_winners[102]
    final_pred = predict_match(
        df, timeline, cache,
        home, away, "2026-07-19",
        as_of=as_of, calibrator=calibrator, injuries=injuries,
    )
    final_pred["home"] = home
    final_pred["away"] = away
    champion = home if final_pred["p_h"] >= final_pred["p_a"] else away

    logger.info(
        f"FINAL {home} vs {away}: "
        f"H={final_pred['p_h']:.0%} D={final_pred['p_d']:.0%} A={final_pred['p_a']:.0%} | "
        f"CHAMPION: {champion}"
    )

    # ---------- GUARDAR ----------
    rows = []
    for w_id, pred in r16_predictions.items():
        rows.append({
            "round": "R16", "match_id": w_id,
            "home": pred["home"], "away": pred["away"],
            "p_h": pred["p_h"], "p_d": pred["p_d"], "p_a": pred["p_a"],
            "predicted_score": pred["predicted_score"],
            "winner": r16_winners[w_id],
        })
    for qf_id, pred in qf_predictions.items():
        rows.append({
            "round": "QF", "match_id": qf_id,
            "home": pred["home"], "away": pred["away"],
            "p_h": pred["p_h"], "p_d": pred["p_d"], "p_a": pred["p_a"],
            "predicted_score": pred["predicted_score"],
            "winner": qf_winners[qf_id],
        })
    for sf_id, pred in sf_predictions.items():
        rows.append({
            "round": "SF", "match_id": sf_id,
            "home": pred["home"], "away": pred["away"],
            "p_h": pred["p_h"], "p_d": pred["p_d"], "p_a": pred["p_a"],
            "predicted_score": pred["predicted_score"],
            "winner": sf_winners[sf_id],
        })
    rows.append({
        "round": "Final", "match_id": 104,
        "home": home, "away": away,
        "p_h": final_pred["p_h"], "p_d": final_pred["p_d"], "p_a": final_pred["p_a"],
        "predicted_score": final_pred["predicted_score"],
        "winner": champion,
    })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(R16_PREDICTIONS_CSV, index=False)
    logger.info(f"\nCSV guardado en {R16_PREDICTIONS_CSV}")

    # ---------- GENERAR MARKDOWN ----------
    md = render_r16_markdown(
        r16_predictions, r16_winners,
        qf_predictions, qf_winners,
        sf_predictions, sf_winners,
        final_pred, champion,
        p87_pred, p88_pred,
        as_of=as_of,
    )
    README_R16.write_bytes(md.encode("utf-8"))
    logger.info(f"Markdown guardado en {README_R16}")


def render_r16_markdown(
    r16_predictions, r16_winners,
    qf_predictions, qf_winners,
    sf_predictions, sf_winners,
    final_pred, champion,
    p87_pred, p88_pred,
    as_of: str,
) -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# WC 2026 - Predicciones R16, QF, SF, Final")
    lines.append("")
    lines.append(f"_Generado {ts} con as_of={as_of} (post-R32)._")
    lines.append("")
    lines.append("Modelo: Poisson + Dixon-Coles + Elo rolling + recent form + features historicas (H2H, momentum, WC history).")
    lines.append("Calibracion: Temperature scaling (T entrenado via LOO sobre 3 mundial).")
    lines.append("")

    # R32 ya jugados (mostrar prediccion del modelo vs realidad)
    lines.append("## R32 resultados (prediccion modelo vs realidad)")
    lines.append("")
    lines.append("| # | Match | H% | D% | A% | Pred modelo | Winner real | OK? |")
    lines.append("|---:|---|---:|---:|---:|---|---|---|")
    # P87: Colombia vs Ghana
    p87_pick = "Colombia" if p87_pred["p_h"] >= max(p87_pred["p_d"], p87_pred["p_a"]) else "Ghana"
    lines.append(
        f"| P87 | Colombia vs Ghana | {p87_pred['p_h']:.0%} | {p87_pred['p_d']:.0%} | {p87_pred['p_a']:.0%} | "
        f"{p87_pred['predicted_score']} | **Colombia** | OK |"
    )
    # P88: Australia vs Egypt
    lines.append(
        f"| P88 | Australia vs Egypt | {p88_pred['p_h']:.0%} | {p88_pred['p_d']:.0%} | {p88_pred['p_a']:.0%} | "
        f"{p88_pred['predicted_score']} | **Egypt** (pens) | {'OK' if p87_pick == 'Colombia' else 'X'} |"
    )
    lines.append("")

    # R16
    lines.append("## R16 - Octavos de final")
    lines.append("")
    lines.append("| # | Match | H% | D% | A% | Pred | Winner |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for w_id in sorted(r16_predictions.keys()):
        p = r16_predictions[w_id]
        lines.append(
            f"| W{w_id} | {p['home']} vs {p['away']} | {p['p_h']:.0%} | {p['p_d']:.0%} | {p['p_a']:.0%} | "
            f"{p['predicted_score']} | **{r16_winners[w_id]}** |"
        )
    lines.append("")

    # QF
    lines.append("## QF - Cuartos de final")
    lines.append("")
    lines.append("| # | Match | H% | D% | A% | Pred | Winner |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for qf_id in sorted(qf_predictions.keys()):
        p = qf_predictions[qf_id]
        lines.append(
            f"| QF{qf_id} | {p['home']} vs {p['away']} | {p['p_h']:.0%} | {p['p_d']:.0%} | {p['p_a']:.0%} | "
            f"{p['predicted_score']} | **{qf_winners[qf_id]}** |"
        )
    lines.append("")

    # SF
    lines.append("## SF - Semifinales")
    lines.append("")
    lines.append("| # | Match | H% | D% | A% | Pred | Winner |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for sf_id in sorted(sf_predictions.keys()):
        p = sf_predictions[sf_id]
        lines.append(
            f"| SF{sf_id} | {p['home']} vs {p['away']} | {p['p_h']:.0%} | {p['p_d']:.0%} | {p['p_a']:.0%} | "
            f"{p['predicted_score']} | **{sf_winners[sf_id]}** |"
        )
    lines.append("")

    # Final
    lines.append("## Final")
    lines.append("")
    lines.append(
        f"| Match | H% | D% | A% | Pred | **CHAMPION** |\n"
        f"|---|---:|---:|---:|---|---|\n"
        f"| {final_pred['home']} vs {final_pred['away']} | "
        f"{final_pred['p_h']:.0%} | {final_pred['p_d']:.0%} | {final_pred['p_a']:.0%} | "
        f"{final_pred['predicted_score']} | **{champion}** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Nota**: Las predicciones R16+ usan `as_of=2026-07-08` (despues del R32) para que el modelo aproveche los resultados del Mundial ya jugados.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
