"""Debug: muestra los componentes de strength para cada equipo en R32."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import get_elo_at, precompute_and_cache
from src.data.historical import load_martj42_csv
from src.data.team_names import OLO_TO_MARTJ
from src.data.wc2026_fixture import generate_group_fixtures
from src.features.historical_features import compute_match_features
from src.features.recent_form import blend_recent_with_historical, compute_recent_form
from src.features.strengths_cache import StrengthsCache
from src.logging_config import get_logger
from src.models import PoissonGoalModel, TeamStrength
from src.simulation.r32_predictions import build_r32_matches

logger = get_logger(__name__)


def _inspect_strengths_for_match(
    home_martj: str, away_martj: str, as_of: str,
    df: pd.DataFrame, cache: StrengthsCache, timeline: dict,
) -> dict:
    """Devuelve los componentes de strength sin predecir."""
    s = get_settings()
    cache.set_elo_snapshot(as_of)
    strengths = cache.get_strengths(
        as_of,
        shrinkage_matches=s.shrinkage_matches,
        min_weighted_matches=s.min_weighted_matches,
    )
    if s.recent_form_n_matches > 0 and s.recent_form_weight > 0:
        train = df[df["date"] < as_of].copy()
        recent = compute_recent_form(
            train, as_of=as_of,
            n_matches=s.recent_form_n_matches,
            min_matches=min(3, s.recent_form_n_matches),
        )
        strengths = blend_recent_with_historical(
            strengths, recent, weight_recent=s.recent_form_weight,
        )

    h_row = strengths[strengths["team"] == home_martj]
    a_row = strengths[strengths["team"] == away_martj]
    if h_row.empty or a_row.empty:
        return {"error": f"Missing strength: home={home_martj} away={away_martj}"}

    h_att_base = float(h_row["attack"].iloc[0])
    h_def_base = float(h_row["defense_vulnerability"].iloc[0])
    a_att_base = float(a_row["attack"].iloc[0])
    a_def_base = float(a_row["defense_vulnerability"].iloc[0])

    h_att_h, h_def_h, a_att_h, a_def_h = compute_match_features(
        df, home_martj, away_martj, as_of, enable=True,
    )
    h_att = h_att_base * h_att_h
    h_def = h_def_base * h_def_h
    a_att = a_att_base * a_att_h
    a_def = a_def_base * a_def_h

    elo_lookup = get_elo_at(timeline, as_of)
    h_elo = elo_lookup.get(home_martj, ORIGINAL_ELO)
    a_elo = elo_lookup.get(away_martj, ORIGINAL_ELO)

    model = PoissonGoalModel(
        draw_penalty_threshold=s.draw_penalty_threshold,
        draw_penalty_strength=s.draw_penalty_strength,
        elo_gap_inflation=s.elo_gap_inflation,
        draw_boost=s.draw_boost,
        league_avg_multiplier=1.0,
    )
    home = TeamStrength(name=home_martj, attack=h_att, defense_vulnerability=h_def)
    away = TeamStrength(name=away_martj, attack=a_att, defense_vulnerability=a_def)
    pred = model.predict(home, away, home_elo=h_elo, away_elo=a_elo)

    return {
        "home": home_martj, "away": away_martj, "as_of": as_of,
        "base": {
            "home_attack": h_att_base, "home_defense": h_def_base,
            "away_attack": a_att_base, "away_defense": a_def_base,
        },
        "features": {
            "home_att_mult": h_att_h, "home_def_mult": h_def_h,
            "away_att_mult": a_att_h, "away_def_mult": a_def_h,
        },
        "final": {
            "home_attack": h_att, "home_defense": h_def,
            "away_attack": a_att, "away_defense": a_def,
        },
        "elo": {"home": h_elo, "away": a_elo, "diff": h_elo - a_elo},
        "probs": {
            "p_h": pred.p_home, "p_d": pred.p_draw, "p_a": pred.p_away,
            "lambda_h": pred.lambda_home, "lambda_a": pred.lambda_away,
        },
    }


def main() -> None:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.parquet")
    logger.info("Cargando datos...")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)

    fixtures = generate_group_fixtures()
    last_played = pd.to_datetime(fixtures[fixtures["played"]]["date"]).max()
    as_of = (last_played + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"as_of = {as_of}")

    def predict_fn(home_olo, away_olo):
        home_martj = OLO_TO_MARTJ.get(home_olo, home_olo)
        away_martj = OLO_TO_MARTJ.get(away_olo, away_olo)
        info = _inspect_strengths_for_match(home_martj, away_martj, as_of, df, cache, timeline)
        if "error" in info:
            return {
                "p_h": 1/3, "p_d": 1/3, "p_a": 1/3,
                "most_likely": (1, 1),
            }
        return {
            "p_h": info["probs"]["p_h"],
            "p_d": info["probs"]["p_d"],
            "p_a": info["probs"]["p_a"],
            "most_likely": (1, 0),
        }

    matches, _, _ = build_r32_matches(fixtures, OLO_TO_MARTJ, predict_fn)
    logger.info("=" * 80)
    logger.info("COMPONENTES DE STRENGTH POR PARTIDO R32")
    logger.info("=" * 80)
    for m in matches:
        home_martj = OLO_TO_MARTJ.get(m.home_team, m.home_team)
        away_martj = OLO_TO_MARTJ.get(m.away_team, m.away_team)
        info = _inspect_strengths_for_match(home_martj, away_martj, as_of, df, cache, timeline)
        if "error" in info:
            logger.info(f"\n#{m.tie_id} {m.home_team} vs {m.away_team}: {info['error']}")
            continue
        b = info["base"]
        f = info["features"]
        fn = info["final"]
        e = info["elo"]
        p = info["probs"]
        logger.info(f"\n#{m.tie_id} {m.home_team} vs {m.away_team}")
        logger.info(
            f"  BASE:    H att={b['home_attack']:.3f} def={b['home_defense']:.3f} | "
            f"A att={b['away_attack']:.3f} def={b['away_defense']:.3f}"
        )
        logger.info(
            f"  FEAT x:  H att_x{f['home_att_mult']:.3f} def_x{f['home_def_mult']:.3f} | "
            f"A att_x{f['away_att_mult']:.3f} def_x{f['away_def_mult']:.3f}"
        )
        logger.info(
            f"  FINAL:   H att={fn['home_attack']:.3f} def={fn['home_defense']:.3f} | "
            f"A att={fn['away_attack']:.3f} def={fn['away_defense']:.3f}"
        )
        logger.info(
            f"  ELO:     H={e['home']:.0f} A={e['away']:.0f} diff={e['diff']:+.0f}"
        )
        logger.info(
            f"  PRED:    H={p['p_h']:.0%} D={p['p_d']:.0%} A={p['p_a']:.0%}  "
            f"(lam H={p['lambda_h']:.2f} A={p['lambda_a']:.2f})"
        )


if __name__ == "__main__":
    main()
