"""Prueba dispersion (Negative Binomial) y otros tweaks del modelo en WC 2026."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.config import get_settings
from src.features.recent_form import compute_recent_form, blend_recent_with_historical
from src.features.strengths_cache import StrengthsCache
from src.models.poisson import PoissonGoalModel
from src.models import TeamStrength
from src.data.elo_timeline import get_elo_at
from src.data.elo import ORIGINAL_ELO

settings = get_settings()
cache_path = Path("data/processed/elo_timeline.parquet")
csv_path = Path("data/raw/martj42_results.csv")
timeline = precompute_and_cache(csv_path, cache_path)
df = load_martj42_csv(csv_path)


def backtest_wc(year, dispersion):
    """Backtest honesto (as_of=match_date) con dispersion variable."""
    wc = df[(df["date"] >= f"{year}-01-01") & (df["date"] < f"{year+1}-01-01")
            & (df["tournament"] == "FIFA World Cup")].copy()
    wc = wc.dropna(subset=["home_goals", "away_goals"])
    if wc.empty:
        return None

    cache = StrengthsCache(df, timeline)
    model = PoissonGoalModel(
        draw_penalty_threshold=settings.draw_penalty_threshold,
        draw_penalty_strength=settings.draw_penalty_strength,
        elo_gap_inflation=settings.elo_gap_inflation,
        draw_boost=settings.draw_boost,
        league_avg_multiplier=settings.world_cup_league_avg_multiplier,
        dispersion=dispersion,
    )

    probs_list = []
    outcomes = []
    for _, m in wc.iterrows():
        match_date = str(m["date"])[:10]
        cache.set_elo_snapshot(match_date)
        strengths = cache.get_strengths(match_date,
                                        shrinkage_matches=settings.shrinkage_matches,
                                        min_weighted_matches=settings.min_weighted_matches)
        if settings.recent_form_n_matches > 0 and settings.recent_form_weight > 0:
            train = df[df["date"] < match_date].copy()
            recent = compute_recent_form(train, as_of=match_date,
                                          n_matches=settings.recent_form_n_matches)
            strengths = blend_recent_with_historical(strengths, recent,
                                                     weight_recent=settings.recent_form_weight)
        h = strengths[strengths["team"] == m["home_team"]]
        a = strengths[strengths["team"] == m["away_team"]]
        if h.empty or a.empty:
            probs_list.append([1/3, 1/3, 1/3])
        else:
            elo_lookup = get_elo_at(timeline, match_date)
            home = TeamStrength(name=m["home_team"], attack=float(h["attack"].iloc[0]),
                                defense_vulnerability=float(h["defense_vulnerability"].iloc[0]))
            away = TeamStrength(name=m["away_team"], attack=float(a["attack"].iloc[0]),
                                defense_vulnerability=float(a["defense_vulnerability"].iloc[0]))
            pred = model.predict(home, away,
                                 home_elo=elo_lookup.get(m["home_team"], ORIGINAL_ELO),
                                 away_elo=elo_lookup.get(m["away_team"], ORIGINAL_ELO))
            probs_list.append([pred.p_home, pred.p_draw, pred.p_away])
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        if hg > ag: outcomes.append(0)
        elif hg < ag: outcomes.append(2)
        else: outcomes.append(1)
    probs = np.array(probs_list)
    outcomes = np.array(outcomes)
    n = len(outcomes)
    # T scaling (current T=0.737)
    T = 0.737
    eps = 1e-9
    probs_c = np.clip(probs, eps, 1-eps)
    probs_c /= probs_c.sum(axis=1, keepdims=True)
    logits = np.log(probs_c)
    logits_T = logits / T
    logits_T -= logits_T.max(axis=1, keepdims=True)
    e = np.exp(logits_T)
    probs = e / e.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    picks = np.argmax(probs, axis=1)
    sign_acc = float((picks == outcomes).mean())
    return brier, sign_acc, n


print(f"{'dispersion':>11s} | {'B 2014':>8s} {'B 2018':>8s} {'B 2022':>8s} {'B 2026':>8s} | {'S 2014':>7s} {'S 2018':>7s} {'S 2022':>7s} {'S 2026':>7s} | {'Avg B':>7s} {'Avg S':>7s}")
print("-" * 110)

for disp in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    bs = []
    ss = []
    for y in [2014, 2018, 2022, 2026]:
        result = backtest_wc(y, disp)
        if result is None:
            continue
        b, s, n = result
        bs.append(b)
        ss.append(s)
    if not bs:
        continue
    avg_b = np.mean(bs)
    avg_s = np.mean(ss)
    bs_str = " ".join(f"{b:.4f}" for b in bs)
    ss_str = " ".join(f"{s*100:.1f}%" for s in ss)
    print(f"{disp:>11.2f} | {bs_str:>40s} | {ss_str:>40s} | {avg_b:.4f} {avg_s*100:.1f}%")
