"""Simulador Monte Carlo del WC 2026 OPTIMIZADO.

Optimizaciones:
1. Precomputa strengths UNA VEZ para todo el torneo (as_of = 2026-06-10).
2. Precalcula probs 1X2 y marcadores para cada par posible (66 pairs de grupos
   + cruces de eliminatorias). El cache de predicciones evita recalcular.
3. Sample de partidos de grupo y eliminatorias es solo rng.choice.

Tiempo objetivo: <30s para 1000 simulaciones.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_settings
from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.data.wc2026_fixture import GROUPS, generate_group_fixtures
from src.evaluation.backtest_elo import get_elo_at
from src.features.recent_form import blend_recent_with_historical, compute_recent_form
from src.features.strengths_cache import StrengthsCache
from src.models import PoissonGoalModel, TeamStrength
from src.simulation.wc2026_bracket import (
    FINAL,
    QUARTER_FINALS,
    ROUND_OF_16,
    ROUND_OF_32,
    SEMI_FINALS,
    SlotKind,
    assign_third_place_slots,
)


# Mapeo Oloraculo -> martj
OLO_TO_MARTJ = {
    "South Korea": "South Korea",
    "Czechia": "Czech Republic",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",
    "Cape Verde": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
}


def _olo_to_martj(name: str) -> str:
    return OLO_TO_MARTJ.get(name, name)


@dataclass
class MatchPrediction:
    p_home: float
    p_draw: float
    p_away: float
    most_likely: tuple[int, int]


class TournamentSimulator:
    """Simulador optimizado del WC 2026."""

    def __init__(
        self,
        df: pd.DataFrame,
        timeline: dict,
        cache: StrengthsCache,
        league_avg_multiplier: float = 1.18,
        as_of: str = "2026-06-10",
    ):
        self.df = df
        self.timeline = timeline
        self.cache = cache
        self.league_avg_multiplier = league_avg_multiplier
        self.as_of = as_of

        # Setup: precomputar strengths una vez
        settings = get_settings()
        self.cache.set_elo_snapshot(as_of)
        self.strengths = cache.get_strengths(
            as_of,
            shrinkage_matches=settings.shrinkage_matches,
            min_weighted_matches=settings.min_weighted_matches,
        )

        # Aplicar recent form
        if settings.recent_form_n_matches > 0 and settings.recent_form_weight > 0:
            train = df[df["date"] < as_of].copy()
            recent = compute_recent_form(
                train,
                as_of=as_of,
                n_matches=settings.recent_form_n_matches,
                min_matches=min(3, settings.recent_form_n_matches),
            )
            self.strengths = blend_recent_with_historical(
                self.strengths, recent, weight_recent=settings.recent_form_weight,
            )

        # Indexar strengths por team (martj)
        self.strength_by_team = self.strengths.set_index("team")

        # Elo lookup
        self.elo_lookup = get_elo_at(timeline, as_of)

        # Modelo
        self.model = PoissonGoalModel(
            draw_penalty_threshold=settings.draw_penalty_threshold,
            draw_penalty_strength=settings.draw_penalty_strength,
            elo_gap_inflation=settings.elo_gap_inflation,
            draw_boost=settings.draw_boost,
            league_avg_multiplier=league_avg_multiplier,
        )

        # Cache de predicciones: (home_martj, away_martj) -> MatchPrediction
        self._pred_cache: dict[tuple[str, str], MatchPrediction] = {}

    def predict(self, home_olo: str, away_olo: str) -> MatchPrediction:
        """Predice un partido. Cachea el resultado."""
        home_martj = _olo_to_martj(home_olo)
        away_martj = _olo_to_martj(away_olo)
        key = (home_martj, away_martj)
        if key in self._pred_cache:
            return self._pred_cache[key]

        try:
            h = self.strength_by_team.loc[home_martj]
            a = self.strength_by_team.loc[away_martj]
        except KeyError:
            self._pred_cache[key] = MatchPrediction(1/3, 1/3, 1/3, (1, 1))
            return self._pred_cache[key]

        home = TeamStrength(
            name=home_martj,
            attack=float(h["attack"]),
            defense_vulnerability=float(h["defense_vulnerability"]),
        )
        away = TeamStrength(
            name=away_martj,
            attack=float(a["attack"]),
            defense_vulnerability=float(a["defense_vulnerability"]),
        )
        home_elo = self.elo_lookup.get(home_martj, ORIGINAL_ELO)
        away_elo = self.elo_lookup.get(away_martj, ORIGINAL_ELO)

        pred = self.model.predict(home, away, home_elo=home_elo, away_elo=away_elo)
        result = MatchPrediction(
            p_home=pred.p_home,
            p_draw=pred.p_draw,
            p_away=pred.p_away,
            most_likely=pred.most_likely_score,
        )
        self._pred_cache[key] = result
        return result

    def sample_group_match(
        self, home_olo: str, away_olo: str, rng: np.random.Generator
    ) -> tuple[str, int, int]:
        """Sample del resultado de un partido de grupo. Retorna (outcome, hg, ag)."""
        pred = self.predict(home_olo, away_olo)
        probs = np.array([pred.p_home, pred.p_draw, pred.p_away])
        probs = probs / probs.sum()
        outcome_idx = rng.choice(3, p=probs)
        outcome = ["H", "D", "A"][outcome_idx]
        hg, ag = pred.most_likely
        if outcome == "H" and hg == ag:
            hg += 1
        elif outcome == "A" and hg == ag:
            ag += 1
        return outcome, hg, ag

    def sample_knockout_match(
        self, home_olo: str, away_olo: str, rng: np.random.Generator
    ) -> str:
        """Sample del ganador de un partido eliminatorio (sin empates)."""
        pred = self.predict(home_olo, away_olo)
        # Remover prob de empate: redistribuir
        probs_no_draw = np.array([pred.p_home, pred.p_away])
        probs_no_draw = probs_no_draw / probs_no_draw.sum()
        idx = rng.choice(2, p=probs_no_draw)
        return home_olo if idx == 0 else away_olo


def simulate_tournament(
    sim: TournamentSimulator,
    fixtures: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """Simula UNA corrida. Retorna dict con campeon, finalists, etc."""
    # --- FASE DE GRUPOS ---
    group_results: dict[str, list] = {g: [] for g in GROUPS}
    for _, fx in fixtures.iterrows():
        if fx["played"]:
            group_results[fx["group"]].append({
                "home": fx["home"],
                "away": fx["away"],
                "home_goals": int(fx["home_score"]),
                "away_goals": int(fx["away_score"]),
            })
        else:
            outcome, hg, ag = sim.sample_group_match(fx["home"], fx["away"], rng)
            group_results[fx["group"]].append({
                "home": fx["home"],
                "away": fx["away"],
                "home_goals": hg,
                "away_goals": ag,
            })

    # Calcular standings
    group_winners = {}
    group_runners_up = {}
    third_info = {}  # grupo -> (team, pts, gd, gf)
    for group, results in group_results.items():
        teams = sorted({r["home"] for r in results} | {r["away"] for r in results})
        pts = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}
        for r in results:
            h, a = r["home"], r["away"]
            hg, ag = r["home_goals"], r["away_goals"]
            pts[h] += 3 if hg > ag else (1 if hg == ag else 0)
            pts[a] += 3 if ag > hg else (1 if hg == ag else 0)
            gf[h] += hg
            ga[h] += ag
            gf[a] += ag
            ga[a] += hg

        standings = sorted(teams, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
        group_winners[group] = standings[0]
        group_runners_up[group] = standings[1]
        if len(standings) >= 3:
            t = standings[2]
            third_info[group] = (t, pts[t], gf[t] - ga[t], gf[t])

    # Top 8 terceros
    sorted_thirds = sorted(third_info.items(), key=lambda x: (-x[1][1], -x[1][2], -x[1][3]))
    qualified_thirds = [g for g, _ in sorted_thirds[:8]]
    third_teams = {g: info[0] for g, info in sorted_thirds[:8]}

    # Asignar terceros a slots
    third_assignments = assign_third_place_slots(qualified_thirds)
    if third_assignments is None:
        return {"error": "No se pudo asignar terceros", "qualified_thirds": qualified_thirds}

    # --- ELIMINATORIAS ---
    def resolve_slot(slot) -> str:
        if slot.kind == SlotKind.GROUP_WINNER:
            return group_winners[slot.group]
        if slot.kind == SlotKind.GROUP_RUNNER_UP:
            return group_runners_up[slot.group]
        if slot.kind == SlotKind.GROUP_THIRD:
            for tie in ROUND_OF_32:
                if tie.id in third_assignments:
                    if (tie.home.kind == SlotKind.GROUP_THIRD
                            and tie.home.third_options == slot.third_options):
                        return third_teams[third_assignments[tie.id]]
                    if (tie.away.kind == SlotKind.GROUP_THIRD
                            and tie.away.third_options == slot.third_options):
                        return third_teams[third_assignments[tie.id]]
            return "?"
        if slot.kind == SlotKind.WINNER_OF:
            return winners[slot.tie_id]
        return "?"

    winners: dict[int, str] = {}

    def play_tie(tie) -> str:
        home = resolve_slot(tie.home)
        away = resolve_slot(tie.away)
        return sim.sample_knockout_match(home, away, rng)

    for tie in ROUND_OF_32:
        winners[tie.id] = play_tie(tie)
    for tie in ROUND_OF_16:
        winners[tie.id] = play_tie(tie)
    for tie in QUARTER_FINALS:
        winners[tie.id] = play_tie(tie)
    for tie in SEMI_FINALS:
        winners[tie.id] = play_tie(tie)
    winners[FINAL.id] = play_tie(FINAL)

    return {
        "champion": winners[FINAL.id],
        "winners": winners,
        "group_winners": group_winners,
        "group_runners_up": group_runners_up,
        "third_teams": third_teams,
        "qualified_thirds": qualified_thirds,
    }


def monte_carlo(
    sim: TournamentSimulator,
    fixtures: pd.DataFrame,
    n_simulations: int = 1000,
) -> dict:
    champion_counts: Counter = Counter()
    reach_r32: Counter = Counter()
    reach_r16: Counter = Counter()
    reach_qf: Counter = Counter()
    reach_sf: Counter = Counter()
    reach_final: Counter = Counter()

    t0 = time.time()
    for i in range(n_simulations):
        rng = np.random.default_rng(42 + i)
        result = simulate_tournament(sim, fixtures, rng)
        if "error" in result:
            continue
        champion_counts[result["champion"]] += 1
        for w in result["group_winners"].values():
            reach_r32[w] += 1
        for r in result["group_runners_up"].values():
            reach_r32[r] += 1
        for t in result["third_teams"].values():
            reach_r32[t] += 1
        for tid in range(73, 89):
            if tid in result["winners"]:
                reach_r16[result["winners"][tid]] += 1
        for tid in range(89, 97):
            if tid in result["winners"]:
                reach_qf[result["winners"][tid]] += 1
        for tid in range(97, 101):
            if tid in result["winners"]:
                reach_sf[result["winners"][tid]] += 1
        for tid in [101, 102]:
            if tid in result["winners"]:
                reach_final[result["winners"][tid]] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_simulations}] {elapsed:.1f}s ({elapsed/(i+1)*n_simulations:.0f}s est.)", flush=True)

    elapsed = time.time() - t0

    all_teams = set(champion_counts) | set(reach_r32) | set(reach_r16) | set(reach_qf) | set(reach_sf) | set(reach_final)
    rows = []
    for team in all_teams:
        rows.append({
            "team": team,
            "champion": champion_counts[team] / n_simulations,
            "final": reach_final[team] / n_simulations,
            "sf": reach_sf[team] / n_simulations,
            "qf": reach_qf[team] / n_simulations,
            "r16": reach_r16[team] / n_simulations,
            "r32": reach_r32[team] / n_simulations,
        })
    df_stats = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
    return {
        "stats": df_stats,
        "elapsed": elapsed,
        "n_simulations": n_simulations,
    }


def main():
    csv_path = Path(r"C:\dev\predictor-mundial\data\raw\martj42_results.csv")
    cache_path = Path(r"C:\dev\predictor-mundial\data\processed\elo_timeline.json")

    print("Cargando datos...", flush=True)
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    print("Construyendo simulador...", flush=True)
    sim = TournamentSimulator(df, timeline, StrengthsCache(df, timeline))
    print(f"  Cache de predicciones listo")

    fixtures = generate_group_fixtures()
    print(f"Fixture: {len(fixtures)} ({fixtures['played'].sum()} FT)")

    print("\nCorriendo 1000 simulaciones Monte Carlo...", flush=True)
    result = monte_carlo(sim, fixtures, n_simulations=1000)

    stats = result["stats"]
    print(f"\nCompletado en {result['elapsed']:.1f}s")
    print(f"\nTop 20 equipos por probabilidad de campeon:\n")
    print(stats.head(20).to_string(index=False))

    stats.to_csv(r"C:\dev\predictor-mundial\wc2026_tournament_probs.csv", index=False)
    print(f"\nGuardado en wc2026_tournament_probs.csv")


if __name__ == "__main__":
    main()
