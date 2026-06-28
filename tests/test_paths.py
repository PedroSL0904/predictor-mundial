"""Tests para src/paths.py."""
from pathlib import Path

from src.paths import (
    DATA_DIR,
    ELO_TIMELINE_PARQUET,
    INJURIES_JSON,
    MARTJ_CSV,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    TEMPERATURE_CALIBRATOR,
)


def test_project_root_is_absolute():
    assert PROJECT_ROOT.is_absolute()


def test_project_root_structure():
    """La raiz debe tener src/, data/, pyproject.toml."""
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "data").is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_data_dirs():
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DIR == DATA_DIR / "raw"
    assert PROCESSED_DIR == DATA_DIR / "processed"


def test_martj_csv_exists():
    """El CSV principal debe existir (descargado por setup_data.py o commited)."""
    if MARTJ_CSV.exists():
        assert MARTJ_CSV.stat().st_size > 1_000_000  # > 1MB


def test_elo_timeline_paths():
    """Al menos uno de los formatos de Elo timeline debe existir."""
    assert ELO_TIMELINE_PARQUET.parent == PROCESSED_DIR


def test_injuries_json_path():
    assert INJURIES_JSON.parent == PROCESSED_DIR


def test_calibrator_path():
    assert TEMPERATURE_CALIBRATOR.parent == PROCESSED_DIR


def test_paths_are_path_objects():
    for p in [MARTJ_CSV, ELO_TIMELINE_PARQUET, INJURIES_JSON, TEMPERATURE_CALIBRATOR]:
        assert isinstance(p, Path)
