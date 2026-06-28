"""Configuración general del proyecto."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    """Configuración cargada desde variables de entorno y/o .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # APIs externas
    odds_api_key: str = ""
    opencode_api_key: str = ""
    understat_enabled: bool = True
    football_data_api_key: str = ""

    # Modelo base
    goal_model_years_window: int = 8
    league_avg_goals: float = 1.35
    league_mean: float = 1.30
    world_cup_league_avg_multiplier: float = 1.18
    max_goals: int = 8
    dispersion: float = 0.0
    rho: float = -0.03

    # Recent form
    recent_form_n_matches: int = 8
    recent_form_weight: float = 0.30
    recent_form_decay_half_life_matches: float = 3.0
    recent_form_min_matches: int = 3
    recent_form_shrink_k: float = 3.0

    # Elo
    elo_k_factor: float = 32.0
    elo_home_advantage: float = 100.0
    elo_gap_threshold: int = 100
    elo_divisor: float = 400.0

    # Anti-sesgo
    draw_penalty_threshold: float = 0.08
    draw_penalty_strength: float = 0.05
    draw_boost: float = 0.10
    elo_gap_inflation: float = 0.30

    # Strengths
    elo_sigma: float = 225.0
    recency_half_life_days: float = 1000.0
    shrinkage_matches: int = 10
    min_weighted_matches: float = 8.0

    # Injuries
    injury_max_attack_penalty: float = 0.20
    injury_max_defense_penalty: float = 0.15
    injury_doubtful_factor: float = 0.5
    injury_min_attack_mult: float = 0.7
    injury_max_defense_mult: float = 1.3

    # Ensemble
    ensemble_enabled: bool = True
    ensemble_weights: list[float] = [0.5, 0.3, 0.2]
    ensemble_bivariate_weight_default: float = 0.3
    ensemble_skellam_weight_default: float = 0.2

    # LLM (OpenCode)
    llm_model: str = "minimax-m3"
    llm_max_tokens_discovery: int = 4000
    llm_max_tokens_enrichment: int = 500
    llm_temperature: float = 0.0

    # Simulación
    n_simulations: int = 10_000
    n_simulations_cli: int = 1000
    simulation_seed: int = 2026

    # Bracket
    r32_tie_ids: tuple[int, int] = (73, 89)
    r16_tie_ids: tuple[int, int] = (89, 97)
    qf_tie_ids: tuple[int, int] = (97, 101)
    sf_tie_ids: tuple[int, int] = (101, 103)
    final_tie_id: int = 104

    # Paths
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR


def get_settings() -> Settings:
    return Settings()
