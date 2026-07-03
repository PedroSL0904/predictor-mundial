"""Genera el README con predicciones actualizadas del WC 2026.

Uso:
  python -m src.cli.wc2026_readme
  # o via entry point: wc2026-readme

Este script:
1. Carga el fixture del WC 2026
2. Para cada partido, genera prediccion con mi sistema
3. Para partidos ya jugados (FT), muestra el resultado real
4. Renderiza un README con tablas por grupo (formato similar a Oloraculo)
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import get_elo_at, precompute_and_cache
from src.data.historical import load_martj42_csv
from src.data.injuries import load_injuries
from src.data.wc2026_fixture import generate_group_fixtures
from src.features.recent_form import blend_recent_with_historical, compute_recent_form
from src.features.strengths_cache import StrengthsCache
from src.logging_config import get_logger
from src.models import PoissonGoalModel, TeamStrength
from src.models.calibration import (
    TemperatureScaler,
    train_temperature_scaler,
)

logger = get_logger(__name__)


def _injury_factors(injuries: dict | None, team_martj: str) -> tuple[float, float]:
    """DEPRECATED: usa src.features.injury_factors.injury_factors.

    Mantenido como shim para backward compat.
    """
    from src.features.injury_factors import injury_factors as _impl
    return _impl(injuries, team_martj)


def predict_match(
    df: pd.DataFrame,
    timeline: dict,
    cache: StrengthsCache,
    home_martj: str,
    away_martj: str,
    match_date: str,
    as_of: str | None = None,
    calibrator: TemperatureScaler | None = None,
    injuries: dict | None = None,
    enable_historical_features: bool = True,
) -> dict:
    """Predice un partido. Retorna dict con p_h, p_d, p_a, predicted_score, top3_scores.

    as_of: fecha de corte para train y snapshot. Si None, usa match_date.
    injuries: dict[martj, TeamInjuries] para ajustar strengths por lesionados.
    enable_historical_features: si True, aplica H2H + momentum + WC history.
    """
    settings = get_settings()
    if as_of is None:
        as_of = match_date

    cache.set_elo_snapshot(as_of)
    strengths = cache.get_strengths(
        as_of,
        shrinkage_matches=settings.shrinkage_matches,
        min_weighted_matches=settings.min_weighted_matches,
    )

    train = df[df["date"] < as_of].copy()
    if settings.recent_form_n_matches > 0 and settings.recent_form_weight > 0:
        recent = compute_recent_form(
            train,
            as_of=match_date,
            n_matches=settings.recent_form_n_matches,
            min_matches=min(3, settings.recent_form_n_matches),
        )
        strengths = blend_recent_with_historical(
            strengths, recent, weight_recent=settings.recent_form_weight,
        )

    h = strengths[strengths["team"] == home_martj]
    a = strengths[strengths["team"] == away_martj]
    if h.empty or a.empty:
        return {
            "p_h": 1/3, "p_d": 1/3, "p_a": 1/3,
            "predicted_score": "1-1",
            "top_scores": [("1-1", 0.10), ("1-0", 0.08), ("0-1", 0.08)],
            "degraded": True,
        }

    # Ajustar por lesionados
    home_att_mult, home_def_mult = _injury_factors(injuries, home_martj)
    away_att_mult, away_def_mult = _injury_factors(injuries, away_martj)

    # Ajustar por features historicas (H2H + momentum + WC history)
    if enable_historical_features:
        from src.features.historical_features import compute_match_features
        h_att_hist, h_def_hist, a_att_hist, a_def_hist = compute_match_features(
            df, home_martj, away_martj, as_of,
        )
        home_att_mult *= h_att_hist
        home_def_mult *= h_def_hist
        away_att_mult *= a_att_hist
        away_def_mult *= a_def_hist

    home = TeamStrength(
        name=home_martj,
        attack=float(h["attack"].iloc[0]) * home_att_mult,
        defense_vulnerability=float(h["defense_vulnerability"].iloc[0]) * home_def_mult,
    )
    away = TeamStrength(
        name=away_martj,
        attack=float(a["attack"].iloc[0]) * away_att_mult,
        defense_vulnerability=float(a["defense_vulnerability"].iloc[0]) * away_def_mult,
    )
    elo_lookup = get_elo_at(timeline, match_date)
    home_elo = elo_lookup.get(home_martj, ORIGINAL_ELO)
    away_elo = elo_lookup.get(away_martj, ORIGINAL_ELO)

    model = PoissonGoalModel(
        draw_penalty_threshold=settings.draw_penalty_threshold,
        draw_penalty_strength=settings.draw_penalty_strength,
        elo_gap_inflation=settings.elo_gap_inflation,
        draw_boost=settings.draw_boost,
        # Mundial 2026 tiene ~17% mas goles que el promedio historico
        # (3.12 vs 2.67 goles/partido en WC 2014-2022). Inflamos λ un 18%.
        # Configurable via settings.world_cup_league_avg_multiplier.
        league_avg_multiplier=settings.world_cup_league_avg_multiplier,
    )
    pred = model.predict(home, away, home_elo=home_elo, away_elo=away_elo)

    # Aplicar calibracion Temperature scaling si esta disponible
    raw_probs = np.array([[pred.p_home, pred.p_draw, pred.p_away]])
    if calibrator is not None and calibrator.fitted:
        cal_probs = calibrator.predict(raw_probs)[0]
    else:
        cal_probs = raw_probs[0]

    return {
        "p_h": float(cal_probs[0]),
        "p_d": float(cal_probs[1]),
        "p_a": float(cal_probs[2]),
        "predicted_score": f"{pred.most_likely_score[0]}-{pred.most_likely_score[1]}",
        "top_scores": [(f"{pred.most_likely_score[0]}-{pred.most_likely_score[1]}", pred.most_likely_score_prob)],
        "degraded": False,
    }


def render_readme(
    predictions_df: pd.DataFrame,
    metrics: dict | None = None,
    tournament_stats: pd.DataFrame | None = None,
    n_simulations: int = 1000,
    sim_elapsed: float = 0.0,
    bracket_analysis: dict | None = None,
) -> str:
    """Renderiza el README con tablas por grupo."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# Predictor Mundial 2026")
    lines.append("")
    lines.append("Predicciones para el Mundial 2026 generadas con el modelo Poisson + Dixon-Coles,")
    lines.append("ponderado por Elo rolling, con ajustes de recent form, draw boost y elo gap inflation.")
    lines.append("")
    lines.append(f"_Generado {timestamp}._")
    lines.append("")

    # Resumen
    n_total = len(predictions_df)
    n_played = predictions_df["played"].sum()
    n_pending = n_total - n_played
    lines.append(f"**{n_total} partidos de fase de grupos** | {n_played} jugados | {n_pending} pendientes")
    lines.append("")

    # Métricas si hay partidos jugados
    if metrics is not None and metrics.get("n_played", 0) > 0:
        lines.append("## Métricas en partidos jugados")
        lines.append("")
        lines.append(f"Partidos evaluados: **{metrics['n_played']}**")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---|---|")
        lines.append(f"| Brier score (1X2) | **{metrics['brier']:.4f}** |")
        lines.append(f"| Log loss | **{metrics['log_loss']:.4f}** |")
        lines.append(f"| Sign accuracy | **{metrics['sign_accuracy']:.1%}** |")
        lines.append(f"| Exact score accuracy | **{metrics['exact_accuracy']:.1%}** |")
        lines.append("")

    # Probabilidades del torneo (Monte Carlo)
    if tournament_stats is not None and not tournament_stats.empty:
        lines.append("## Probabilidades del torneo (Monte Carlo)")
        lines.append("")
        lines.append(
            f"Simulacion: {n_simulations} corridas del torneo completo "
            f"(fase de grupos + R32 + R16 + QF + SF + Final). "
            f"Respetando los {n_played} partidos ya jugados. "
            f"Tiempo: {sim_elapsed:.1f}s."
        )
        lines.append("")
        lines.append("Top 16 por probabilidad de campeon:")
        lines.append("")
        lines.append("| Equipo | Campeon | Final | SF | QF | R16 | R32 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in tournament_stats.head(16).iterrows():
            lines.append(
                f"| {r['team']} | {r['champion']:.1%} | {r['final']:.1%} | "
                f"{r['sf']:.1%} | {r['qf']:.1%} | {r['r16']:.1%} | {r['r32']:.1%} |"
            )
        lines.append("")

    # Llave completa de eliminatorias (analisis de los 32 partidos de la llave)
    if bracket_analysis is not None:
        from src.simulation.bracket_analysis import format_bracket_tree
        bracket_md = format_bracket_tree(bracket_analysis)
        # Split multi-line string y agregar a lines
        lines.extend(bracket_md.split("\n"))

    # Tablas por grupo
    lines.append("## Grupos")
    lines.append("")

    for group in sorted(predictions_df["group"].unique()):
        group_df = predictions_df[predictions_df["group"] == group].copy()
        group_letter = group.split()[-1] if "Group" in group else group

        # Ordenar: FT primero (por fecha), luego pendientes (por fecha)
        group_df["sort_key"] = group_df.apply(
            lambda r: (0 if r["played"] else 1, r["date"] or "9999"), axis=1
        )
        group_df = group_df.sort_values("sort_key")

        lines.append(f"### Group {group_letter}")
        lines.append("")
        lines.append("| Match | Status | Pick / Result | H | D | A |")
        lines.append("|---|---|---|---:|---:|---:|")
        for _, row in group_df.iterrows():
            home = row["home"]
            away = row["away"]
            ph = row["p_h"]
            pd_ = row["p_d"]
            pa = row["p_a"]
            ph_str = f"{ph:.0%}" if not pd.isna(ph) else "-"
            pd_str = f"{pd_:.0%}" if not pd.isna(pd_) else "-"
            pa_str = f"{pa:.0%}" if not pd.isna(pa) else "-"

            if row["played"]:
                result = f"{int(row['home_score'])}-{int(row['away_score'])}"
                pred_score = row.get("predicted_score", "?")
                # Top-pick: el outcome con mayor prob
                if not pd.isna(ph):
                    top_pick = "H" if ph == max(ph, pd_, pa) else ("D" if pd_ == max(ph, pd_, pa) else "A")
                else:
                    top_pick = "?"
                actual = "H" if row["home_score"] > row["away_score"] else ("A" if row["home_score"] < row["away_score"] else "D")
                mark = "OK" if top_pick == actual else "X"
                pick_str = f"**{result}**<br><sub>Pred: {pred_score} -&gt; {top_pick} ({mark})</sub>"
                status = "FT"
            else:
                pred_score = row.get("predicted_score", "?")
                pick_str = pred_score
                if row["date"]:
                    status = f"{row['date']}"
                else:
                    status = "TBD"

            lines.append(f"| {home} vs {away} | {status} | {pick_str} | {ph_str} | {pd_str} | {pa_str} |")
        lines.append("")

    lines.append("<!-- predictor:snapshots:end -->")
    lines.append("")

    return "\n".join(lines)


def compute_metrics(predictions_df: pd.DataFrame) -> dict:
    """Calcula metricas en partidos FT."""
    played = predictions_df[predictions_df["played"]].copy()
    if played.empty:
        return {}

    outcomes = []
    for _, r in played.iterrows():
        if r["home_score"] > r["away_score"]:
            outcomes.append(0)
        elif r["home_score"] < r["away_score"]:
            outcomes.append(2)
        else:
            outcomes.append(1)
    outcomes = np.array(outcomes)
    probs = played[["p_h", "p_d", "p_a"]].values

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcomes] = 1
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())

    eps = 1e-9
    logloss = float(-np.log(np.maximum(probs[np.arange(len(probs)), outcomes], eps)).mean())

    picks = np.argmax(probs, axis=1)
    sign_acc = float((picks == outcomes).mean())

    # Exact score accuracy
    played["actual_score"] = played.apply(
        lambda r: f"{int(r['home_score'])}-{int(r['away_score'])}", axis=1
    )
    exact_acc = float((played["predicted_score"] == played["actual_score"]).mean())

    return {
        "n_played": len(played),
        "brier": brier,
        "log_loss": logloss,
        "sign_accuracy": sign_acc,
        "exact_accuracy": exact_acc,
    }


def _compute_as_of(fixtures: pd.DataFrame) -> str:
    """Calcula la fecha de corte para entrenar y predecir.

    Usa el dia siguiente al ultimo partido FT para que el modelo aproveche
    los resultados del Mundial ya jugados al predecir partidos futuros.
    """
    if fixtures["played"].any():
        last_played = pd.to_datetime(fixtures[fixtures["played"]]["date"]).max()
        return (last_played + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return "2026-06-10"


def _compute_as_of_from_csv(df: pd.DataFrame, default: str = "2026-06-10") -> str:
    """Calcula as_of a partir del CSV completo (incluye R32, R16, etc).

    Usa el dia siguiente al ultimo partido FT en el CSV, no solo los
    partidos de fase de grupos.
    """
    played = df.dropna(subset=["home_goals", "away_goals"])
    if played.empty:
        return default
    last = pd.to_datetime(played["date"]).max()
    return (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _load_data():
    """Carga CSV, timeline de Elo y StrengthsCache."""
    from src.paths import ELO_TIMELINE_JSON, MARTJ_CSV

    logger.info("Cargando datos...")
    timeline = precompute_and_cache(MARTJ_CSV, ELO_TIMELINE_JSON)
    df = load_martj42_csv(MARTJ_CSV)
    cache = StrengthsCache(df, timeline)
    return df, timeline, cache


def _train_calibrator() -> TemperatureScaler:
    """Entrena y guarda el calibrador de temperature scaling via LOO."""
    from src.paths import TEMPERATURE_CALIBRATOR

    logger.info("Entrenando calibrador (Temperature scaling LOO)...")
    calibrator = train_temperature_scaler()
    logger.info(f"  T_optimo: {calibrator.T_:.3f} (T<1 = comprimir, T>1 = expandir)")
    cal_path = TEMPERATURE_CALIBRATOR
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(cal_path)
    logger.info(f"  Guardado en {cal_path}")
    return calibrator


def _predict_all_matches(
    fixtures: pd.DataFrame,
    df: pd.DataFrame,
    timeline: dict,
    cache: StrengthsCache,
    as_of: str,
    calibrator: TemperatureScaler,
    injuries_dict: dict,
) -> pd.DataFrame:
    """Predice todos los partidos de fase de grupos. Retorna DataFrame con probs."""
    logger.info(f"Prediciendo {len(fixtures)} partidos (as_of={as_of})...")
    if injuries_dict:
        n_out = sum(len(ti.out) for ti in injuries_dict.values())
        logger.info(f"  Lesionados: {len(injuries_dict)} equipos, {n_out} jugadores out")

    rows = []
    t0 = time.time()
    total = len(fixtures)
    for i, (_, fx) in enumerate(fixtures.iterrows()):
        if i % 10 == 0:
            logger.info(f"  [{i}/{total}]")
        match_date = fx["date"] if fx["played"] else (fx["date"] or "2026-06-15")
        try:
            pred = predict_match(
                df, timeline, cache,
                fx["home_martj"], fx["away_martj"],
                match_date,
                as_of=as_of,
                calibrator=calibrator,
                injuries=injuries_dict,
            )
        except Exception as e:
            logger.info(f"Error prediciendo {fx['home']} vs {fx['away']}: {e}")
            pred = {"p_h": np.nan, "p_d": np.nan, "p_a": np.nan,
                    "predicted_score": "?", "top_scores": [], "degraded": True}
        rows.append({**fx.to_dict(), **pred})

    pred_df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    logger.info(f"Predicciones listas en {elapsed:.1f}s")
    return pred_df


def _run_simulations(
    df: pd.DataFrame,
    timeline: dict,
    cache: StrengthsCache,
    fixtures: pd.DataFrame,
    as_of: str,
    calibrator: TemperatureScaler,
    injuries: dict,
) -> dict:
    """Corre Monte Carlo del torneo + analisis de llave."""
    from src.simulation.bracket_analysis import analyze_bracket
    from src.simulation.wc2026_simulate import (
        TournamentSimulator,
        extract_r32_actual_winners,
        monte_carlo,
    )

    logger.info("Corriendo 1000 simulaciones Monte Carlo del torneo...")
    if injuries:
        n_teams = len(injuries)
        n_out = sum(len(ti.out) for ti in injuries.values())
        logger.info(f"  Lesionados cargados: {n_teams} equipos, {n_out} jugadores out")

    # Detectar R32 ya jugados: si hay resultados en el CSV post-2026-07-01,
    # usarlos como winners reales (no simular R32 aleatoriamente).
    r32_actual = extract_r32_actual_winners(df)
    if r32_actual:
        logger.info(f"  R32 jugados: {len(r32_actual)}/16 winners reales usados")
    else:
        logger.info("  R32 no han empezado: simulacion completa desde R32")

    sim = TournamentSimulator(
        df, timeline, cache, as_of=as_of,
        calibrator=calibrator,
        injuries=injuries,
    )
    mc_result = monte_carlo(
        sim, fixtures, n_simulations=1000,
        r32_actual_winners=r32_actual if r32_actual else None,
    )
    tournament_stats = mc_result["stats"]
    logger.info(f"  Monte Carlo en {mc_result['elapsed']:.1f}s")
    logger.info(f"  Top 3: {tournament_stats.head(3)['team'].tolist()}")

    logger.info(f"Analizando llave completa ({mc_result['n_simulations']} simulaciones)...")
    bracket_analysis = analyze_bracket(
        sim, fixtures, n_simulations=mc_result["n_simulations"]
    )
    return {
        "mc_result": mc_result,
        "tournament_stats": tournament_stats,
        "bracket_analysis": bracket_analysis,
        "r32_actual_winners": r32_actual,
    }


def _save_outputs(
    pred_df: pd.DataFrame,
    tournament_stats: pd.DataFrame,
    metrics: dict,
    mc_result: dict,
    bracket_analysis: dict,
) -> None:
    """Genera y guarda README + CSVs."""
    from src.paths import (
        PREDICTIONS_CSV,
        README_WC2026,
        TOURNAMENT_PROBS_CSV,
    )

    readme = render_readme(
        pred_df, metrics, tournament_stats,
        n_simulations=mc_result["n_simulations"],
        sim_elapsed=mc_result["elapsed"],
        bracket_analysis=bracket_analysis,
    )
    # UTF-8 sin BOM para compatibilidad maxima
    README_WC2026.write_bytes(readme.encode("utf-8"))
    logger.info(f"README guardado en {README_WC2026}")

    pred_df[["group", "date", "home", "away", "played", "home_score", "away_score",
             "predicted_score", "p_h", "p_d", "p_a"]].to_csv(PREDICTIONS_CSV, index=False)
    logger.info(f"CSV guardado en {PREDICTIONS_CSV}")

    tournament_stats.to_csv(TOURNAMENT_PROBS_CSV, index=False)
    logger.info(f"Tournament probs guardado en {TOURNAMENT_PROBS_CSV}")


def main() -> None:
    """Pipeline principal: load → train → predict → simulate → save."""
    logger.info("Generando fixture WC 2026...")
    fixtures = generate_group_fixtures()

    df, timeline, cache = _load_data()
    calibrator = _train_calibrator()
    as_of = _compute_as_of(fixtures)
    injuries = load_injuries()

    pred_df = _predict_all_matches(
        fixtures, df, timeline, cache, as_of, calibrator, injuries,
    )

    metrics = compute_metrics(pred_df)
    if metrics:
        logger.info(
            f"Métricas: Brier={metrics['brier']:.4f}, "
            f"Sign={metrics['sign_accuracy']:.1%}, n={metrics['n_played']}"
        )

    sim_outputs = _run_simulations(
        df, timeline, cache, fixtures, as_of, calibrator, injuries,
    )

    _save_outputs(
        pred_df,
        sim_outputs["tournament_stats"],
        metrics,
        sim_outputs["mc_result"],
        sim_outputs["bracket_analysis"],
    )


if __name__ == "__main__":
    main()
