"""Comparacion predictor-mundial vs Oloraculo en WC 2026.

Extrae las predicciones de Oloraculo del README y genera las mismas
predicciones con nuestro sistema. Compara lado a lado.

Para partidos ya jugados (FT), calcula Brier y sign accuracy.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(r"C:\dev\predictor-mundial").resolve()))

from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv, normalize_team_name
from src.features.recent_form import blend_recent_with_historical, compute_recent_form
from src.features.strengths_cache import StrengthsCache
from src.models import PoissonGoalModel, TeamStrength


# Mapeo directo de nombre Oloraculo -> nombre EXACTO en martj42_results.csv
# (sin pasar por normalize_team_name que rompe el matching)
OLO_TO_MARTJ = {
    "South Korea": "South Korea",
    "South Africa": "South Africa",
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "Saudi Arabia": "Saudi Arabia",
    "Congo DR": "DR Congo",
    "Ivory Coast": "Ivory Coast",
    "United States": "United States",
    "Scotland": "Scotland",
    "England": "England",
    "Wales": "Wales",
    "Northern Ireland": "Northern Ireland",
    "Republic of Ireland": "Republic of Ireland",
    "Iran": "Iran",
    "North Korea": "North Korea",
    "Curaçao": "Curaçao",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
}


def parse_oloraculo_readme(readme_path: Path) -> pd.DataFrame:
    """Extrae las predicciones de los partidos de grupo del README de Oloraculo."""
    text = readme_path.read_text(encoding="utf-8")
    rows = []
    current_group = None

    for line in text.split("\n"):
        m = re.match(r"<summary><strong>(Group [A-L])</strong></summary>", line)
        if m:
            current_group = m.group(1)
            continue

        if current_group is None or "<img" not in line or " vs " not in line:
            continue

        # Patron: alt=""> HOME vs <img ... alt=""> AWAY | ...
        m_teams = re.search(
            r'alt="">\s*([^<]+?)\s*vs\s*<img[^>]+alt="">\s*([^<|]+?)\s*\|',
            line,
        )
        if not m_teams:
            continue
        home = m_teams.group(1).strip()
        away = m_teams.group(2).strip()

        # Status: FT o fecha
        m_status = re.search(r"\|\s*(FT|\w{3}\s+\d+\s+\d+:\d+\s+UTC)\s*\|", line)
        status = m_status.group(1) if m_status else ""

        # Result si FT: **2-0**
        m_result = re.search(r"\*\*(\d+-\d+)\*\*", line)
        result = m_result.group(1) if m_result else ""

        # Pick (Prediction:)
        m_pick = re.search(r"Prediction:\s*([^<|]+?)(?:</sub>|\||$)", line)
        pick = m_pick.group(1).strip() if m_pick else ""

        # Probabilidades H/D/A: tres % al final de la linea
        # Split por | y tomar los ultimos 4 campos (Why, H%, D%, A%)
        parts = [p.strip() for p in line.split("|")]
        m_pct = []
        for p in parts:
            mm = re.match(r"^(\d+)\s*%$", p)
            if mm:
                m_pct.append(int(mm.group(1)))
        if len(m_pct) >= 3:
            ph = m_pct[-3] / 100
            pd_ = m_pct[-2] / 100
            pa = m_pct[-1] / 100
        else:
            ph = pd_ = pa = float("nan")

        rows.append({
            "group": current_group,
            "home": home,
            "away": away,
            "status": status,
            "result": result,
            "olo_pick": pick,
            "olo_ph": ph,
            "olo_pd": pd_,
            "olo_pa": pa,
        })
    return pd.DataFrame(rows)


def get_elo_at(timeline, as_of: str) -> dict:
    candidates = [d for d in timeline if d <= as_of]
    if not candidates:
        return {}
    return timeline[max(candidates)]


def predict_match_my_system(
    df: pd.DataFrame,
    timeline: dict,
    cache: StrengthsCache,
    home_name: str,
    away_name: str,
    match_date: str,
    home_elo: float | None = None,
    away_elo: float | None = None,
) -> tuple[float, float, float]:
    """Predice un partido con mi sistema. Retorna (p_h, p_d, p_a)."""
    from src.config import get_settings
    settings = get_settings()

    cache.set_elo_snapshot(match_date)
    strengths = cache.get_strengths(
        match_date,
        shrinkage_matches=settings.shrinkage_matches,
        min_weighted_matches=settings.min_weighted_matches,
    )

    train = df[df["date"] < match_date].copy()
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

    h = strengths[strengths["team"] == home_name]
    a = strengths[strengths["team"] == away_name]
    if h.empty or a.empty:
        return 0.33, 0.34, 0.33

    home = TeamStrength(
        name=home_name,
        attack=float(h["attack"].iloc[0]),
        defense_vulnerability=float(h["defense_vulnerability"].iloc[0]),
    )
    away = TeamStrength(
        name=away_name,
        attack=float(a["attack"].iloc[0]),
        defense_vulnerability=float(a["defense_vulnerability"].iloc[0]),
    )

    if home_elo is None or away_elo is None:
        elo_lookup = get_elo_at(timeline, match_date)
        home_elo = elo_lookup.get(home_name, ORIGINAL_ELO)
        away_elo = elo_lookup.get(away_name, ORIGINAL_ELO)

    model = PoissonGoalModel(
        draw_penalty_threshold=settings.draw_penalty_threshold,
        draw_penalty_strength=settings.draw_penalty_strength,
        elo_gap_inflation=settings.elo_gap_inflation,
        draw_boost=settings.draw_boost,
    )
    pred = model.predict(home, away, home_elo=home_elo, away_elo=away_elo)
    return pred.p_home, pred.p_draw, pred.p_away


def main() -> None:
    readme = Path(r"C:\dev\Oloraculo\README.md")
    olo_df = parse_oloraculo_readme(readme)
    print(f"Partidos parseados de Oloraculo README: {len(olo_df)}")
    print(olo_df[["group", "home", "away", "status", "olo_pick", "olo_ph", "olo_pd", "olo_pa"]].head(10))

    # Cargar mi sistema
    csv_path = Path(r"C:\dev\predictor-mundial\data\raw\martj42_results.csv")
    cache_path = Path(r"C:\dev\predictor-mundial\data\processed\elo_timeline.json")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)

    # Para cada partido, predecir con mi sistema
    rows = []
    # Construir set de equipos en el dataset para matching flexible
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    for _, row in olo_df.iterrows():
        home_olo = row["home"]
        away_olo = row["away"]
        # Mapeo directo Oloraculo -> nombre exacto en martj42_results.csv
        home_martj = OLO_TO_MARTJ.get(home_olo, home_olo)
        away_martj = OLO_TO_MARTJ.get(away_olo, away_olo)
        # Si no existe en el dataset, intentar matching flexible
        if home_martj not in all_teams:
            for t in all_teams:
                if t.lower().replace(" ", "").replace("ç", "c").replace("ã", "a") == \
                   home_martj.lower().replace(" ", "").replace("ç", "c").replace("ã", "a"):
                    home_martj = t
                    break
        if away_martj not in all_teams:
            for t in all_teams:
                if t.lower().replace(" ", "").replace("ç", "c").replace("ã", "a") == \
                   away_martj.lower().replace(" ", "").replace("ç", "c").replace("ã", "a"):
                    away_martj = t
                    break

        # Mapear fecha: partidos ya jugados o por jugar
        # Usamos 2026-06-15 como fecha generica (inicio del WC)
        match_date = "2026-06-15"

        try:
            ph, pd_, pa = predict_match_my_system(
                df, timeline, cache,
                home_martj, away_martj, match_date,
            )
        except Exception as e:
            ph, pd_, pa = float("nan"), float("nan"), float("nan")
            print(f"Error prediciendo {home_olo} vs {away_olo}: {e}")

        rows.append({
            **row.to_dict(),
            "home_martj": home_martj,
            "away_martj": away_martj,
            "my_ph": ph,
            "my_pd": pd_,
            "my_pa": pa,
        })

    res = pd.DataFrame(rows)
    res.to_csv(r"C:\dev\predictor-mundial\compare_wc2026.csv", index=False)
    print(f"\nComparativa guardada en compare_wc2026.csv")
    print(f"Partidos: {len(res)}")
    print(f"Partidos FT: {(res['status'] == 'FT').sum()}")
    print(f"Partidos con prediccion disponible: {res['my_ph'].notna().sum()}")


if __name__ == "__main__":
    main()
