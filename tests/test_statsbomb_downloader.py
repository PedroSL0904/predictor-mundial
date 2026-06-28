"""Tests del downloader de StatsBomb (sin red: solo funciones de transformacion)."""
from __future__ import annotations

from src.data.statsbomb_downloader import extract_match_xg


def test_extract_xg_simple() -> None:
    """1 shot del home con xg=0.5."""
    events = [
        {
            "type": {"name": "Shot"},
            "team": {"name": "Spain"},
            "shot": {"statsbomb_xg": 0.5},
        }
    ]
    h_xg, a_xg = extract_match_xg(events, "Spain", "France")
    assert abs(h_xg - 0.5) < 1e-9
    assert a_xg == 0.0


def test_extract_xg_both_teams() -> None:
    events = [
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}, "shot": {"statsbomb_xg": 0.3}},
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}, "shot": {"statsbomb_xg": 0.2}},
        {"type": {"name": "Shot"}, "team": {"name": "France"}, "shot": {"statsbomb_xg": 0.4}},
    ]
    h_xg, a_xg = extract_match_xg(events, "Spain", "France")
    assert abs(h_xg - 0.5) < 1e-9
    assert abs(a_xg - 0.4) < 1e-9


def test_extract_xg_ignores_non_shots() -> None:
    """Solo 'Shot' events cuentan. Pass, Foul, etc son ignorados."""
    events = [
        {"type": {"name": "Pass"}, "team": {"name": "Spain"}, "shot": {"statsbomb_xg": 0.99}},
        {"type": {"name": "Foul"}, "team": {"name": "France"}, "shot": {"statsbomb_xg": 0.99}},
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}, "shot": {"statsbomb_xg": 0.1}},
    ]
    h_xg, a_xg = extract_match_xg(events, "Spain", "France")
    assert h_xg == 0.1
    assert a_xg == 0.0


def test_extract_xg_missing_xg_field() -> None:
    """Si el shot no tiene xg, no se cuenta (no crashea)."""
    events = [
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}},
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}, "shot": {"statsbomb_xg": 0.2}},
    ]
    h_xg, a_xg = extract_match_xg(events, "Spain", "France")
    assert abs(h_xg - 0.2) < 1e-9


def test_extract_xg_unknown_team() -> None:
    """Teams en events que no son home/away son ignorados."""
    events = [
        {"type": {"name": "Shot"}, "team": {"name": "Referee"}, "shot": {"statsbomb_xg": 0.99}},
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}, "shot": {"statsbomb_xg": 0.1}},
    ]
    h_xg, a_xg = extract_match_xg(events, "Spain", "France")
    assert h_xg == 0.1
    assert a_xg == 0.0


def test_extract_xg_empty_events() -> None:
    h_xg, a_xg = extract_match_xg([], "Spain", "France")
    assert h_xg == 0.0
    assert a_xg == 0.0


def test_extract_xg_legacy_field() -> None:
    """Soporta el campo legacy shot_statsbomb_xg ademas de shot.statsbomb_xg."""
    events = [
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}, "shot_statsbomb_xg": 0.3},
    ]
    h_xg, a_xg = extract_match_xg(events, "Spain", "France")
    assert abs(h_xg - 0.3) < 1e-9


def test_extract_xg_none_xg_treated_as_zero() -> None:
    """shot.statsbomb_xg=None (no NoneType crash)."""
    events = [
        {"type": {"name": "Shot"}, "team": {"name": "Spain"}, "shot": {"statsbomb_xg": None}},
    ]
    h_xg, a_xg = extract_match_xg(events, "Spain", "France")
    assert h_xg == 0.0
