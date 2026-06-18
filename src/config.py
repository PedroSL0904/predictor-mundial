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
    understat_enabled: bool = True
    football_data_api_key: str = ""

    # Modelo
    goal_model_years_window: int = 8
    recent_form_n_matches: int = 8
    recent_form_weight: float = 0.30
    elo_k_factor: float = 32.0
    elo_home_advantage: float = 100.0

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

    # Modelo
    dispersion: float = 0.0

    # Simulación
    n_simulations: int = 10_000
    simulation_seed: int = 2026

    # Paths
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR


def get_settings() -> Settings:
    return Settings()
