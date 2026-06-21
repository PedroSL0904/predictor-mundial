"""Genera el README con predicciones actualizadas del WC 2026.

Uso:
  python -m src.cli.wc2026_readme

Este script:
1. Carga el fixture del WC 2026
2. Para cada partido, genera prediccion con mi sistema
3. Para partidos ya jugados (FT), muestra el resultado real
4. Renderiza un README con tablas por grupo (formato similar a Oloraculo)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_settings
from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.data.wc2026_fixture import generate_group_fixtures
from src.features.recent_form import blend_recent_with_historical, compute_recent_form
from src.features.strengths_cache import StrengthsCache
from src.models import PoissonGoalModel, TeamStrength


def get_elo_at(timeline: dict, as_of: str) -> dict:
    candidates = [d for d in timeline if d <= as_of]
    if not candidates:
        return {}
    return timeline[max(candidates)]


def predict_match(
    df: pd.DataFrame,
    timeline: dict,
    cache: StrengthsCache,
    home_martj: str,
    away_martj: str,
    match_date: str,
    as_of: str | None = None,
) -> dict:
    """Predice un partido. Retorna dict con p_h, p_d, p_a, predicted_score, top3_scores.

    as_of: fecha de corte para train y snapshot. Si None, usa match_date.
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

    home = TeamStrength(
        name=home_martj,
        attack=float(h["attack"].iloc[0]),
        defense_vulnerability=float(h["defense_vulnerability"].iloc[0]),
    )
    away = TeamStrength(
        name=away_martj,
        attack=float(a["attack"].iloc[0]),
        defense_vulnerability=float(a["defense_vulnerability"].iloc[0]),
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
        league_avg_multiplier=1.18,
    )
    pred = model.predict(home, away, home_elo=home_elo, away_elo=away_elo)

    return {
        "p_h": pred.p_home,
        "p_d": pred.p_draw,
        "p_a": pred.p_away,
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


def main() -> None:
    csv_path = Path(r"C:\dev\predictor-mundial\data\raw\martj42_results.csv")
    cache_path = Path(r"C:\dev\predictor-mundial\data\processed\elo_timeline.json")

    print("Cargando datos...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)

    print("Generando fixture WC 2026...", flush=True)
    fixtures = generate_group_fixtures()

    # Predecir cada partido
    # Usamos una fecha de corte comun (1 dia antes del inicio del WC) para
    # todos los partidos, simulando que las predicciones se generan ANTES del
    # torneo. Esto evita leakage de partidos ya jugados.
    AS_OF = "2026-06-10"
    print(f"Prediciendo {len(fixtures)} partidos (as_of={AS_OF})...", flush=True)
    rows = []
    t0 = time.time()
    for i, (_, fx) in enumerate(fixtures.iterrows()):
        if i % 10 == 0:
            print(f"  [{i}/{len(fixtures)}]", flush=True)
        match_date = fx["date"] if fx["played"] else (fx["date"] or "2026-06-15")
        try:
            pred = predict_match(
                df, timeline, cache,
                fx["home_martj"], fx["away_martj"],
                match_date,
                as_of=AS_OF,
            )
        except Exception as e:
            print(f"Error prediciendo {fx['home']} vs {fx['away']}: {e}")
            pred = {"p_h": np.nan, "p_d": np.nan, "p_a": np.nan,
                    "predicted_score": "?", "top_scores": [], "degraded": True}

        rows.append({
            **fx.to_dict(),
            **pred,
        })

    pred_df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"Predicciones listas en {elapsed:.1f}s")

    # Métricas
    metrics = compute_metrics(pred_df)
    if metrics:
        print(f"Métricas: Brier={metrics['brier']:.4f}, "
              f"Sign={metrics['sign_accuracy']:.1%}, n={metrics['n_played']}")

    # Simulacion Monte Carlo del torneo completo
    print("Corriendo 1000 simulaciones Monte Carlo del torneo...", flush=True)
    from src.simulation.wc2026_simulate import TournamentSimulator, monte_carlo
    sim = TournamentSimulator(df, timeline, cache, as_of=AS_OF)
    mc_result = monte_carlo(sim, fixtures, n_simulations=1000)
    tournament_stats = mc_result["stats"]
    print(f"  Monte Carlo en {mc_result['elapsed']:.1f}s")
    print(f"  Top 3: {tournament_stats.head(3)['team'].tolist()}")

    # Generar README
    readme = render_readme(
        pred_df, metrics, tournament_stats,
        n_simulations=mc_result["n_simulations"],
        sim_elapsed=mc_result["elapsed"],
    )
    readme_path = Path(r"C:\dev\predictor-mundial\WC2026_README.md")
    # Usar UTF-8 sin BOM para compatibilidad maxima
    readme_path.write_bytes(readme.encode("utf-8"))
    print(f"README guardado en {readme_path}")

    # También guardar CSV con predicciones
    csv_out = Path(r"C:\dev\predictor-mundial\wc2026_predictions.csv")
    pred_df[["group", "date", "home", "away", "played", "home_score", "away_score",
             "predicted_score", "p_h", "p_d", "p_a"]].to_csv(csv_out, index=False)
    print(f"CSV guardado en {csv_out}")

    # Guardar CSV con probabilidades del torneo
    tournament_stats.to_csv(
        r"C:\dev\predictor-mundial\wc2026_tournament_probs.csv", index=False
    )
    print(f"Tournament probs guardado en wc2026_tournament_probs.csv")


if __name__ == "__main__":
    main()
