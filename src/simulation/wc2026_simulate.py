"""Simulador Monte Carlo del WC 2026 OPTIMIZADO.

Optimizaciones:
1. Precomputa strengths UNA VEZ para todo el torneo (as_of = 2026-06-10).
2. Precalcula probs 1X2 y marcadores para cada par posible (66 pairs de grupos
   + cruces de eliminatorias). El cache de predicciones evita recalcular.
3. Sample de partidos de grupo y eliminatorias es solo rng.choice.

Tiempo objetivo: <30s para 1000 simulaciones.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.elo import ORIGINAL_ELO
from src.data.elo_timeline import get_elo_at, precompute_and_cache
from src.data.historical import load_martj42_csv
from src.data.team_names import OLO_TO_MARTJ
from src.data.wc2026_fixture import GROUPS, generate_group_fixtures
from src.features.recent_form import blend_recent_with_historical, compute_recent_form
from src.features.strengths_cache import StrengthsCache
from src.logging_config import get_logger
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

logger = get_logger(__name__)


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
        calibrator = None,
        injuries: dict | None = None,
        model: object | None = None,
        enable_historical_features: bool = True,
    ):
        self.df = df
        self.timeline = timeline
        self.cache = cache
        self.league_avg_multiplier = league_avg_multiplier
        self.as_of = as_of
        self.injuries = injuries or {}  # dict[martj_name, TeamInjuries]
        self.enable_historical_features = enable_historical_features

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

        # Modelo (Sprint A7: acepta cualquier modelo con .predict() interface.
        # Default: PoissonGoalModel para backward compat.)
        if model is None:
            self.model = PoissonGoalModel(
                draw_penalty_threshold=settings.draw_penalty_threshold,
                draw_penalty_strength=settings.draw_penalty_strength,
                elo_gap_inflation=settings.elo_gap_inflation,
                draw_boost=settings.draw_boost,
                league_avg_multiplier=league_avg_multiplier,
            )
        else:
            self.model = model

        # Calibrador (Temperature scaling)
        self.calibrator = calibrator

        # Cache de predicciones: (home_martj, away_martj) -> MatchPrediction
        self._pred_cache: dict[tuple[str, str], MatchPrediction] = {}

    def _injury_factors(self, team_martj: str) -> tuple[float, float]:
        """DEPRECATED: usa src.features.injury_factors.injury_factors.

        Mantenido como shim para backward compat.
        """
        from src.features.injury_factors import injury_factors as _impl
        return _impl(self.injuries, team_martj)

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

        # Aplicar ajustes por lesionados
        home_attack_mult, home_def_mult = self._injury_factors(home_martj)
        away_attack_mult, away_def_mult = self._injury_factors(away_martj)

        # Aplicar features historicas (H2H, momentum, WC history)
        if self.enable_historical_features:
            from src.features.historical_features import compute_match_features
            h_att_hist, h_def_hist, a_att_hist, a_def_hist = compute_match_features(
                self.df, home_martj, away_martj, self.as_of,
            )
        else:
            h_att_hist = h_def_hist = a_att_hist = a_def_hist = 1.0
        home_attack_mult *= h_att_hist
        home_def_mult *= h_def_hist
        away_attack_mult *= a_att_hist
        away_def_mult *= a_def_hist

        home = TeamStrength(
            name=home_martj,
            attack=float(h["attack"]) * home_attack_mult,
            defense_vulnerability=float(h["defense_vulnerability"]) * home_def_mult,
        )
        away = TeamStrength(
            name=away_martj,
            attack=float(a["attack"]) * away_attack_mult,
            defense_vulnerability=float(a["defense_vulnerability"]) * away_def_mult,
        )
        home_elo = self.elo_lookup.get(home_martj, ORIGINAL_ELO)
        away_elo = self.elo_lookup.get(away_martj, ORIGINAL_ELO)

        pred = self.model.predict(home, away, home_elo=home_elo, away_elo=away_elo)

        # Aplicar calibracion
        if self.calibrator is not None and self.calibrator.fitted:
            raw = np.array([[pred.p_home, pred.p_draw, pred.p_away]])
            cal = self.calibrator.predict(raw)[0]
            p_h, p_d, p_a = float(cal[0]), float(cal[1]), float(cal[2])
        else:
            p_h, p_d, p_a = pred.p_home, pred.p_draw, pred.p_away

        result = MatchPrediction(
            p_home=p_h,
            p_draw=p_d,
            p_away=p_a,
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
    r32_actual_winners: dict[int, str] | None = None,
    r16_actual_winners: dict[int, str] | None = None,
) -> dict:
    """Simula UNA corrida. Retorna dict con campeon, finalists, etc.

    Args:
        sim: TournamentSimulator con as_of y strengths precomputados.
        fixtures: DataFrame de partidos de grupo (72 filas).
        rng: numpy random generator.
        r32_actual_winners: dict[tie_id, team_name] con los winners REALES del
            R32 (tie_id 73-88). Si se proporciona, NO simula R32: usa estos
            winners directamente y arranca el Monte Carlo desde R16.
            Si es None, simula R32 aleatoriamente (modo "what-if").
        r16_actual_winners: dict[tie_id, team_name] con los winners REALES del
            R16 (tie_id 89-96). Si se proporciona, NO simula R16: usa estos
            winners directamente y arranca el Monte Carlo desde QF.
    """
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

    # --- R32: usar winners reales si se proporcionan ---
    if r32_actual_winners:
        # Asignar winners reales (R32 ya se jugo, no simulamos)
        for tie in ROUND_OF_32:
            if tie.id in r32_actual_winners:
                winners[tie.id] = r32_actual_winners[tie.id]
            else:
                # TBD (e.g., P87 o P88 no jugados) - simular
                winners[tie.id] = play_tie(tie)
    else:
        for tie in ROUND_OF_32:
            winners[tie.id] = play_tie(tie)
    # --- R16: usar winners reales si se proporcionan ---
    if r16_actual_winners:
        for tie in ROUND_OF_16:
            if tie.id in r16_actual_winners:
                winners[tie.id] = r16_actual_winners[tie.id]
            else:
                winners[tie.id] = play_tie(tie)
    else:
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
        "third_assignments": third_assignments,
    }


def monte_carlo(
    sim: TournamentSimulator,
    fixtures: pd.DataFrame,
    n_simulations: int = 1000,
    r32_actual_winners: dict[int, str] | None = None,
    r16_actual_winners: dict[int, str] | None = None,
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
        result = simulate_tournament(sim, fixtures, rng,
                                     r32_actual_winners=r32_actual_winners,
                                     r16_actual_winners=r16_actual_winners)
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
            logger.info(f"  [{i+1}/{n_simulations}] {elapsed:.1f}s ({elapsed/(i+1)*n_simulations:.0f}s est.)")

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


def extract_r32_actual_winners(
    df: pd.DataFrame,
    olo_to_martj: dict[str, str] | None = None,
) -> dict[int, str]:
    """Extrae winners reales del R32 del CSV.

    Para cada partido con date entre 2026-07-01 y 2026-07-08 en
    tournament "FIFA World Cup" y con resultado, determina el winner
    y lo mapea a (tie_id, team_name_olo).

    Args:
        df: DataFrame del CSV martj42.
        olo_to_martj: opcional, mapping martj->olo (no usado actualmente,
            los nombres en el CSV son martj y se usan directo).

    Returns:
        dict[tie_id (73-88), team_name (martj/olo)].
        Solo incluye partidos jugados.
    """

    # Construir mapping: (home_martj, away_martj) -> tie_id
    # Para esto, primero determinamos los matchups del R32 desde
    # los standings de grupos (no del CSV, porque el CSV tiene los
    # partidos pero no los tie_ids).
    # El matchup depende de los standings finales, que ya conocemos.

    # Por simplicidad, hardcodeamos los winners conocidos del WC 2026.
    # Esto se actualiza manualmente a medida que se juegan los partidos.
    # (Alternativa: derivar del bracket code, pero requiere resolver
    # los standings primero.)

    r32 = df[(df["date"] >= "2026-07-01") & (df["date"] <= "2026-07-08")
             & (df["tournament"] == "FIFA World Cup")].copy()
    r32 = r32.dropna(subset=["home_goals", "away_goals"])

    if r32.empty:
        return {}

    # Mapping hardcoded de (home, away) -> tie_id
    # Basado en el bracket FIFA y los standings finales del WC 2026.
    # Si un partido nuevo se juega, agregar aqui.
    KNOWN_MATCHUPS: dict[tuple[str, str], int] = {
        ("South Africa", "Canada"): 73,
        ("Germany", "Paraguay"): 74,
        ("Netherlands", "Morocco"): 75,
        ("Brazil", "Japan"): 76,
        ("France", "Sweden"): 77,
        ("Ivory Coast", "Norway"): 78,
        ("Mexico", "Ecuador"): 79,
        ("England", "DR Congo"): 80,
        ("United States", "Bosnia and Herzegovina"): 81,
        ("Belgium", "Senegal"): 82,
        ("Portugal", "Croatia"): 83,
        ("Spain", "Austria"): 84,
        ("Switzerland", "Algeria"): 85,
        ("Argentina", "Cape Verde"): 86,
        ("Colombia", "Ghana"): 87,
        ("Australia", "Egypt"): 88,
    }

    # Overrides para partidos decididos en penalties (CSV solo tiene score 90 min).
    # Winner real difiere del que sugiere el score.
    PENALTY_OVERRIDES: dict[tuple[str, str], str] = {
        ("Netherlands", "Morocco"): "Morocco",  # P75, MAR 4-3 pens
        ("Australia", "Egypt"): "Egypt",         # P88, EGY 4-2 pens
    }

    winners: dict[int, str] = {}
    for _, m in r32.iterrows():
        key = (m["home_team"], m["away_team"])
        tie_id = KNOWN_MATCHUPS.get(key)
        if tie_id is None:
            # Partido no esperado (o nuevo). Skipear.
            continue
        # Override por penalties si aplica
        if key in PENALTY_OVERRIDES:
            winners[tie_id] = PENALTY_OVERRIDES[key]
            continue
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        winner_martj = m["home_team"] if hg > ag else (
            m["away_team"] if hg < ag else m["home_team"]  # empate: usar home por convencion
        )
        winners[tie_id] = winner_martj
    return winners


def extract_r16_actual_winners(
    df: pd.DataFrame,
) -> dict[int, str]:
    """Extrae winners reales del R16 del CSV.

    Similar a extract_r32_actual_winners pero para partidos del R16
    (fechas 2026-07-04 a 2026-07-10). Mapea (home, away) -> tie_id (89-96).

    Returns:
        dict[tie_id (89-96), team_name (martj)].
        Solo incluye partidos jugados.
    """
    r16 = df[(df["date"] >= "2026-07-04") & (df["date"] <= "2026-07-10")
             & (df["tournament"] == "FIFA World Cup")].copy()
    r16 = r16.dropna(subset=["home_goals", "away_goals"])

    if r16.empty:
        return {}

    # Mapping hardcoded R16 (home, away) -> tie_id
    # Si se juega un nuevo R16, agregar aqui.
    KNOWN_MATCHUPS_R16: dict[tuple[str, str], int] = {
        ("Paraguay", "France"): 89,
        ("Canada", "Morocco"): 90,
        ("Brazil", "Norway"): 91,
        ("Mexico", "England"): 92,
        ("Portugal", "Spain"): 93,
        ("United States", "Belgium"): 94,
        ("Switzerland", "Colombia"): 95,
        ("Argentina", "Egypt"): 96,
    }

    winners: dict[int, str] = {}
    for _, m in r16.iterrows():
        key = (m["home_team"], m["away_team"])
        tie_id = KNOWN_MATCHUPS_R16.get(key)
        if tie_id is None:
            # Partido no esperado (o nuevo). Skipear.
            continue
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        winner_martj = m["home_team"] if hg > ag else (
            m["away_team"] if hg < ag else m["home_team"]
        )
        winners[tie_id] = winner_martj
    return winners


def main():
    from src.paths import (
        ELO_TIMELINE_JSON,
        MARTJ_CSV,
        TEMPERATURE_CALIBRATOR,
        TOURNAMENT_PROBS_CSV,
    )
    csv_path = MARTJ_CSV
    cache_path = ELO_TIMELINE_JSON
    cal_path = TEMPERATURE_CALIBRATOR

    logger.info("Cargando datos...")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)

    # Cargar calibrador si existe
    calibrator = None
    if cal_path.exists():
        from src.models.calibration import TemperatureScaler
        calibrator = TemperatureScaler.load(cal_path)
        logger.info(f"  Calibrador: T={calibrator.T_:.3f}")
    else:
        logger.info("  Calibrador: no encontrado (sin calibrar)")

    logger.info("Construyendo simulador...")
    sim = TournamentSimulator(
        df, timeline, StrengthsCache(df, timeline),
        calibrator=calibrator,
    )
    logger.info("  Cache de predicciones listo")

    fixtures = generate_group_fixtures()
    logger.info(f"Fixture: {len(fixtures)} ({fixtures['played'].sum()} FT)")

    logger.info("\nCorriendo 1000 simulaciones Monte Carlo...")
    result = monte_carlo(sim, fixtures, n_simulations=1000)

    stats = result["stats"]
    logger.info(f"\nCompletado en {result['elapsed']:.1f}s")
    logger.info("\nTop 20 equipos por probabilidad de campeon:\n")
    logger.info(stats.head(20).to_string(index=False))

    stats.to_csv(TOURNAMENT_PROBS_CSV, index=False)
    logger.info(f"\nGuardado en {TOURNAMENT_PROBS_CSV}")


if __name__ == "__main__":
    main()
