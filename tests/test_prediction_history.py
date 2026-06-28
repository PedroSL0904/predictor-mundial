"""Tests del tracking persistente de predicciones."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.prediction_history import PredictionHistory
from src.domain import MatchPrediction


@pytest.fixture
def tmp_history(tmp_path: Path) -> PredictionHistory:
    """History backed by a temp file."""
    return PredictionHistory(path=tmp_path / "history.csv")


def _make_pred(
    home: str = "Argentina",
    away: str = "France",
    p_h: float = 0.5,
    p_d: float = 0.3,
    p_a: float = 0.2,
    score: tuple[int, int] = (2, 1),
) -> MatchPrediction:
    return MatchPrediction(
        home_team=home,
        away_team=away,
        model="test_model",
        p_home=p_h,
        p_draw=p_d,
        p_away=p_a,
        lambda_home=1.5,
        lambda_away=1.0,
        most_likely_score=score,
        most_likely_score_prob=0.1,
    )


def test_record_creates_entry(tmp_history: PredictionHistory) -> None:
    pred = _make_pred()
    tmp_history.record(pred, "2026-06-15", "Argentina", "France")
    df = tmp_history.get_history()
    assert len(df) == 1
    assert df.iloc[0]["home"] == "Argentina"
    assert df.iloc[0]["p_home"] == 0.5


def test_record_persists_to_disk(tmp_path: Path) -> None:
    """Una instancia nueva debe poder leer lo que escribio otra."""
    path = tmp_path / "history.csv"
    h1 = PredictionHistory(path=path)
    h1.record(_make_pred(), "2026-06-15", "Argentina", "France")
    h2 = PredictionHistory(path=path)
    df = h2.get_history()
    assert len(df) == 1
    assert df.iloc[0]["away"] == "France"


def test_record_overwrites_existing(tmp_history: PredictionHistory) -> None:
    """Re-record del mismo match debe pisar el anterior."""
    tmp_history.record(_make_pred(p_h=0.5), "2026-06-15", "Argentina", "France")
    tmp_history.record(_make_pred(p_h=0.7), "2026-06-15", "Argentina", "France")
    df = tmp_history.get_history()
    assert len(df) == 1
    assert df.iloc[0]["p_home"] == 0.7


def test_record_preserves_outcome_on_overwrite(tmp_history: PredictionHistory) -> None:
    """Al re-record un match con outcome, el outcome se preserva."""
    tmp_history.record(_make_pred(), "2026-06-15", "Argentina", "France")
    tmp_history.update_outcome("2026-06-15", "Argentina", "France", 2, 1)
    tmp_history.record(_make_pred(p_h=0.6), "2026-06-15", "Argentina", "France")
    df = tmp_history.get_history()
    assert len(df) == 1
    assert df.iloc[0]["p_home"] == 0.6
    assert df.iloc[0]["outcome"] == "H"  # preservado
    assert df.iloc[0]["home_score"] == 2


def test_record_different_models_separate_entries(tmp_history: PredictionHistory) -> None:
    """Dos modelos distintos para el mismo match son entries separados."""
    pred1 = _make_pred()
    pred1.model = "poisson"
    pred2 = _make_pred()
    pred2.model = "skellam"
    tmp_history.record(pred1, "2026-06-15", "Argentina", "France")
    tmp_history.record(pred2, "2026-06-15", "Argentina", "France")
    df = tmp_history.get_history()
    assert len(df) == 2
    assert set(df["model"]) == {"poisson", "skellam"}


def test_update_outcome_sets_scores(tmp_history: PredictionHistory) -> None:
    tmp_history.record(_make_pred(), "2026-06-15", "Argentina", "France")
    n = tmp_history.update_outcome("2026-06-15", "Argentina", "France", 2, 1)
    assert n == 1
    df = tmp_history.get_with_outcome()
    assert len(df) == 1
    assert df.iloc[0]["outcome"] == "H"
    assert df.iloc[0]["home_score"] == 2
    assert df.iloc[0]["away_score"] == 1


def test_update_outcome_no_match(tmp_history: PredictionHistory) -> None:
    """Si no hay match, update_outcome retorna 0."""
    n = tmp_history.update_outcome("2026-06-15", "Argentina", "France", 2, 1)
    assert n == 0


def test_update_outcome_draw(tmp_history: PredictionHistory) -> None:
    tmp_history.record(_make_pred(), "2026-06-15", "Spain", "Portugal")
    tmp_history.update_outcome("2026-06-15", "Spain", "Portugal", 1, 1)
    df = tmp_history.get_history()
    assert df.iloc[0]["outcome"] == "D"


def test_update_outcome_away_win(tmp_history: PredictionHistory) -> None:
    tmp_history.record(_make_pred(), "2026-06-15", "Spain", "Portugal")
    tmp_history.update_outcome("2026-06-15", "Spain", "Portugal", 0, 2)
    df = tmp_history.get_history()
    assert df.iloc[0]["outcome"] == "A"


def test_get_unmatched_returns_pending(tmp_history: PredictionHistory) -> None:
    pred = _make_pred()
    tmp_history.record(pred, "2026-06-15", "Argentina", "France")
    tmp_history.record(pred, "2026-06-20", "Brazil", "Germany")
    tmp_history.update_outcome("2026-06-15", "Argentina", "France", 2, 1)
    unmatched = tmp_history.get_unmatched()
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["home"] == "Brazil"


def test_get_metrics_perfect_prediction(tmp_history: PredictionHistory) -> None:
    """Predicciones perfectas → brier=0, sign=1."""
    pred = _make_pred(p_h=1.0, p_d=0.0, p_a=0.0)
    tmp_history.record(pred, "2026-06-15", "Argentina", "France")
    tmp_history.update_outcome("2026-06-15", "Argentina", "France", 2, 1)
    m = tmp_history.get_metrics()
    assert m["n"] == 1
    assert m["brier"] < 0.01
    assert m["sign_acc"] == 1.0


def test_get_metrics_empty(tmp_history: PredictionHistory) -> None:
    """Sin outcomes, get_metrics devuelve {}."""
    pred = _make_pred()
    tmp_history.record(pred, "2026-06-15", "Argentina", "France")
    m = tmp_history.get_metrics()
    assert m == {}


def test_clear_empties_history(tmp_history: PredictionHistory) -> None:
    pred = _make_pred()
    tmp_history.record(pred, "2026-06-15", "Argentina", "France")
    assert len(tmp_history.get_history()) == 1
    tmp_history.clear()
    assert len(tmp_history.get_history()) == 0


def test_handles_corrupt_file(tmp_path: Path) -> None:
    """Si el archivo esta corrupto, se inicia vacio."""
    path = tmp_path / "history.csv"
    path.write_text("not a valid csv\nwith bad data")
    h = PredictionHistory(path=path)
    assert len(h.get_history()) == 0
