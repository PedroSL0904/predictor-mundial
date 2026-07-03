"""Prueba rapida de diferentes parametros draw_boost / draw_penalty en WC 2026."""
from src.cli.wc2026_readme import predict_match, _load_data, _compute_as_of_from_csv
from src.data.injuries import load_injuries
from src.models.calibration import TemperatureScaler
from src.config import get_settings
import numpy as np
import pandas as pd

df, timeline, cache = _load_data()
injuries = load_injuries()
as_of = _compute_as_of_from_csv(df)

wc = df[(df["date"] >= "2026-06-01") & (df["tournament"] == "FIFA World Cup")].copy()
wc = wc.dropna(subset=["home_goals", "away_goals"])

# Precalcular outcomes
outcomes_list = []
for _, m in wc.iterrows():
    hg, ag = int(m["home_goals"]), int(m["away_goals"])
    if hg > ag: outcomes_list.append(0)
    elif hg < ag: outcomes_list.append(2)
    else: outcomes_list.append(1)
outcomes = np.array(outcomes_list)
n = len(outcomes)


def evaluate(draw_boost, draw_penalty_strength, draw_penalty_threshold,
             elo_gap_inflation, T):
    """Evalua con params dados. Retorna (brier, sign_acc, n_draws_picked, draw_acc)."""
    from src.models.poisson import PoissonGoalModel
    from src.cli.wc2026_readme import predict_match
    from src.features.recent_form import compute_recent_form, blend_recent_with_historical
    from src.data.elo_timeline import get_elo_at
    from src.data.elo import ORIGINAL_ELO
    from src.models import TeamStrength

    settings = get_settings()
    cache.set_elo_snapshot(as_of)
    strengths = cache.get_strengths(as_of, shrinkage_matches=settings.shrinkage_matches,
                                   min_weighted_matches=settings.min_weighted_matches)
    train = df[df["date"] < as_of].copy()
    if settings.recent_form_n_matches > 0 and settings.recent_form_weight > 0:
        recent = compute_recent_form(train, as_of=as_of, n_matches=settings.recent_form_n_matches)
        strengths = blend_recent_with_historical(strengths, recent, weight_recent=settings.recent_form_weight)

    elo_lookup = get_elo_at(timeline, as_of)
    model = PoissonGoalModel(
        draw_penalty_threshold=draw_penalty_threshold,
        draw_penalty_strength=draw_penalty_strength,
        elo_gap_inflation=elo_gap_inflation,
        draw_boost=draw_boost,
        league_avg_multiplier=settings.world_cup_league_avg_multiplier,
    )

    # T scaling simple
    eps = 1e-9
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    probs_list = []
    for _, m in wc.iterrows():
        h = strengths[strengths["team"] == m["home_team"]]
        a = strengths[strengths["team"] == m["away_team"]]
        if h.empty or a.empty:
            probs_list.append([1/3, 1/3, 1/3])
            continue
        home = TeamStrength(name=m["home_team"], attack=float(h["attack"].iloc[0]),
                            defense_vulnerability=float(h["defense_vulnerability"].iloc[0]))
        away = TeamStrength(name=m["away_team"], attack=float(a["attack"].iloc[0]),
                            defense_vulnerability=float(a["defense_vulnerability"].iloc[0]))
        pred = model.predict(home, away,
                             home_elo=elo_lookup.get(m["home_team"], ORIGINAL_ELO),
                             away_elo=elo_lookup.get(m["away_team"], ORIGINAL_ELO))
        probs_list.append([pred.p_home, pred.p_draw, pred.p_away])

    probs = np.array(probs_list)
    # Apply temperature scaling properly
    eps = 1e-9
    probs_c = np.clip(probs, eps, 1-eps)
    probs_c /= probs_c.sum(axis=1, keepdims=True)
    logits = np.log(probs_c)
    # Softmax 2D
    logits_T = logits / T
    logits_T -= logits_T.max(axis=1, keepdims=True)
    e = np.exp(logits_T)
    probs = e / e.sum(axis=1, keepdims=True)

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcomes] = 1
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    picks = np.argmax(probs, axis=1)
    sign_acc = float((picks == outcomes).mean())
    n_draws_picked = (picks == 1).sum()
    draw_acc = float((picks[outcomes == 1] == 1).mean()) if (outcomes == 1).sum() > 0 else 0
    return brier, sign_acc, n_draws_picked, draw_acc


# Baseline (current params)
print(f"{'draw_boost':>11s} {'penalty':>8s} {'threshold':>10s} {'elo_infl':>8s} {'T':>5s} | {'Brier':>8s} {'Sign%':>6s} {'#D_pick':>7s} {'D_acc%':>7s}")
print("-" * 100)

# Current
settings = get_settings()
b, s, nd, da = evaluate(
    settings.draw_boost, settings.draw_penalty_strength,
    settings.draw_penalty_threshold, settings.elo_gap_inflation, 0.737,
)
print(f"{settings.draw_boost:>11.2f} {settings.draw_penalty_strength:>8.3f} {settings.draw_penalty_threshold:>10.2f} {settings.elo_gap_inflation:>8.2f} {0.737:>5.3f} | {b:>8.4f} {s*100:>5.1f}% {nd:>7d} {da*100:>6.1f}%  (current)")

# Variaciones
for db in [0.80, 1.00, 1.20, 1.50]:
    for dp in [0.01]:
        for di in [0.20, 0.25]:
            for dt in [0.12, 0.15, 0.20]:
                b, s, nd, da = evaluate(db, dp, dt, di, 0.737)
                print(f"{db:>11.2f} {dp:>8.3f} {dt:>10.2f} {di:>8.2f} {0.737:>5.3f} | {b:>8.4f} {s*100:>5.1f}% {nd:>7d} {da*100:>6.1f}%")

print()
print("=== Con T mayor (suavizar) ===")
for T in [1.0, 1.3, 1.5, 1.8, 2.0]:
    b, s, nd, da = evaluate(0.30, 0.02, 0.12, 0.20, T)
    print(f"{0.30:>11.2f} {0.02:>8.3f} {0.12:>10.2f} {0.20:>8.2f} {T:>5.3f} | {b:>8.4f} {s*100:>5.1f}% {nd:>7d} {da*100:>6.1f}%")
