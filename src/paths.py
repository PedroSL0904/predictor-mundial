"""Centralizacion de paths del proyecto.

Todos los archivos del proyecto deben importar paths desde aca en vez de
hardcodear rutas absolutas. Esto permite que el proyecto sea portable
(Linux/Mac, otras PCs, otras instalaciones).

Convenciones:
- PROJECT_ROOT: raiz del proyecto
- DATA_DIR: data/ con CSV fuente y archivos procesados
- OUTPUT_DIR: raiz del proyecto (donde van WC2026_README.md, CSVs de output)

Uso:
    from src.paths import MARTJ_CSV, INJURIES_JSON

    df = pd.read_csv(MARTJ_CSV)
"""
from __future__ import annotations

from pathlib import Path

# Raiz del proyecto: 2 niveles arriba de src/paths.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Archivos fuente
MARTJ_CSV: Path = RAW_DIR / "martj42_results.csv"
STATSBOMB_XG: Path = RAW_DIR / "statsbomb_xg.json"

# Archivos procesados (cache)
ELO_TIMELINE_PARQUET: Path = PROCESSED_DIR / "elo_timeline.parquet"
ELO_TIMELINE_JSON: Path = PROCESSED_DIR / "elo_timeline.json"  # legacy
TEMPERATURE_CALIBRATOR: Path = PROCESSED_DIR / "temperature_calibrator.json"
INJURIES_JSON: Path = PROCESSED_DIR / "injuries.json"
PREDICTIONS_HISTORY: Path = PROCESSED_DIR / "predictions_history.csv"

# Output en raiz del proyecto
README_WC2026: Path = PROJECT_ROOT / "WC2026_README.md"
README_R32: Path = PROJECT_ROOT / "WC2026_R32.md"
PREDICTIONS_CSV: Path = PROJECT_ROOT / "wc2026_predictions.csv"
TOURNAMENT_PROBS_CSV: Path = PROJECT_ROOT / "wc2026_tournament_probs.csv"
R32_PREDICTIONS_CSV: Path = PROJECT_ROOT / "wc2026_r32_predictions.csv"
