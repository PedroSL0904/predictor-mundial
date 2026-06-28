"""Tests para src/config.py."""
from src.config import Settings, get_settings


def test_settings_equivalent():
    """Cada llamada devuelve una instancia equivalente (no necesariamente la misma)."""
    s1 = get_settings()
    s2 = get_settings()
    # Equivalentes en valores clave
    assert s1.elo_k_factor == s2.elo_k_factor
    assert s1.draw_boost == s2.draw_boost
    assert s1.world_cup_league_avg_multiplier == s2.world_cup_league_avg_multiplier


def test_default_values():
    s = Settings()
    assert s.elo_k_factor == 32.0
    assert s.elo_home_advantage == 100.0
    assert s.draw_boost == 0.10
    assert s.elo_gap_inflation == 0.30
    assert s.recent_form_n_matches == 8
    assert s.recent_form_weight == 0.30
    assert s.shrinkage_matches == 10
    assert s.min_weighted_matches == 8.0
    assert s.elo_sigma == 225.0
    assert s.recency_half_life_days == 1000.0


def test_new_injury_settings():
    s = Settings()
    assert s.injury_max_attack_penalty == 0.20
    assert s.injury_max_defense_penalty == 0.15
    assert s.injury_doubtful_factor == 0.5
    assert s.injury_min_attack_mult == 0.7
    assert s.injury_max_defense_mult == 1.3


def test_new_model_settings():
    s = Settings()
    assert s.league_avg_goals == 1.35
    assert s.league_mean == 1.30
    assert s.world_cup_league_avg_multiplier == 1.18
    assert s.max_goals == 8
    assert s.rho == -0.03
    assert s.elo_gap_threshold == 100
    assert s.elo_divisor == 400.0


def test_new_ensemble_settings():
    s = Settings()
    assert s.ensemble_enabled is True
    assert s.ensemble_weights == [0.5, 0.3, 0.2]
    assert s.ensemble_bivariate_weight_default == 0.3
    assert s.ensemble_skellam_weight_default == 0.2


def test_new_llm_settings():
    s = Settings()
    assert s.llm_model == "minimax-m3"
    assert s.llm_max_tokens_discovery == 4000
    assert s.llm_max_tokens_enrichment == 500
    assert s.llm_temperature == 0.0


def test_new_bracket_settings():
    s = Settings()
    assert s.r32_tie_ids == (73, 89)
    assert s.r16_tie_ids == (89, 97)
    assert s.qf_tie_ids == (97, 101)
    assert s.sf_tie_ids == (101, 103)
    assert s.final_tie_id == 104


def test_recent_form_settings():
    s = Settings()
    assert s.recent_form_decay_half_life_matches == 3.0
    assert s.recent_form_min_matches == 3
    assert s.recent_form_shrink_k == 3.0


def test_simulation_settings():
    s = Settings()
    assert s.n_simulations == 10_000
    assert s.n_simulations_cli == 1000
    assert s.simulation_seed == 2026
