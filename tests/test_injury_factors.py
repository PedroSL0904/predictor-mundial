"""Tests para src/features/injury_factors.py (unica implementacion)."""
from src.data.injuries import PlayerStatus, TeamInjuries
from src.features.injury_factors import injury_factors


def test_no_injuries_returns_1_1():
    assert injury_factors(None, "Argentina") == (1.0, 1.0)
    assert injury_factors({}, "Argentina") == (1.0, 1.0)


def test_empty_team_injuries_returns_1_1():
    ti = TeamInjuries(team="Argentina", out=[], doubtful=[])
    assert injury_factors({"Argentina": ti}, "Argentina") == (1.0, 1.0)


def test_unknown_team_returns_1_1():
    ti = TeamInjuries(team="Argentina", out=[PlayerStatus(name="X", reason="injury", position="FWD", importance=0.5)])
    assert injury_factors({"Argentina": ti}, "Brazil") == (1.0, 1.0)


def test_single_out_reduces_attack():
    ti = TeamInjuries(
        team="Argentina",
        out=[PlayerStatus(name="Messi", reason="injury", position="FWD", importance=0.5)],
        doubtful=[],
    )
    attack_mult, defense_mult = injury_factors({"Argentina": ti}, "Argentina")
    assert attack_mult < 1.0  # reduccion
    assert defense_mult == 1.0  # sin cambio en defensa


def test_single_out_increases_defense_vulnerability():
    ti = TeamInjuries(
        team="Argentina",
        out=[PlayerStatus(name="Romero", reason="injury", position="DEF", importance=0.5)],
        doubtful=[],
    )
    attack_mult, defense_mult = injury_factors({"Argentina": ti}, "Argentina")
    assert attack_mult == 1.0  # sin cambio en ataque
    assert defense_mult > 1.0  # aumenta vulnerabilidad


def test_doubtful_penalty_smaller_than_out():
    """Doubtful aplica 50% del penalty de OUT."""
    out_ti = TeamInjuries(
        team="A", out=[PlayerStatus(name="X", reason="injury", position="FWD", importance=0.6)],
        doubtful=[],
    )
    doubt_ti = TeamInjuries(
        team="A", out=[], doubtful=[PlayerStatus(name="X", reason="injury", position="FWD", importance=0.6)],
    )
    _, out_defense = injury_factors({"A": out_ti}, "A")
    _, doubt_defense = injury_factors({"A": doubt_ti}, "A")
    # out produce mayor reduccion de attack que doubt
    out_attack, _ = injury_factors({"A": out_ti}, "A")
    doubt_attack, _ = injury_factors({"A": doubt_ti}, "A")
    assert out_attack < doubt_attack


def test_min_attack_mult_floor():
    """Attack no puede bajar de 0.7 incluso con muchos lesionados."""
    ti = TeamInjuries(
        team="A",
        out=[
            PlayerStatus(name=f"X{i}", reason="injury", position="FWD", importance=1.0)
            for i in range(10)
        ],
        doubtful=[],
    )
    attack_mult, _ = injury_factors({"A": ti}, "A")
    assert attack_mult >= 0.7  # minimo


def test_max_defense_mult_cap():
    """Defense no puede subir de 1.3 incluso con muchos lesionados."""
    ti = TeamInjuries(
        team="A",
        out=[
            PlayerStatus(name=f"X{i}", reason="injury", position="DEF", importance=1.0)
            for i in range(10)
        ],
        doubtful=[],
    )
    _, defense_mult = injury_factors({"A": ti}, "A")
    assert defense_mult <= 1.3  # maximo
