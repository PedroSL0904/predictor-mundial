"""Datos de lesionados y suspendidos para el WC 2026.

Estructura:
    {
        "team_olo_name": {
            "out": [
                {"player": "Name", "reason": "injury/suspension", "expected_return": "YYYY-MM-DD or null"}
            ],
            "doubtful": [...],
            "source": "wikipedia|manual|api",
            "last_updated": "YYYY-MM-DD"
        }
    }

Los pesos de importancia por jugador se estiman desde statsbomb xG o
minutos jugados si están disponibles. El ajuste al modelo es:
    attack * (1 - sum_importance_out * 0.5)
    defense_vulnerability * (1 + sum_importance_defenders_out * 0.3)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

INJURIES_PATH = Path(r"C:\dev\predictor-mundial\data\processed\injuries.json")


@dataclass
class PlayerStatus:
    name: str
    reason: str  # "injury", "suspension", "illness"
    expected_return: Optional[str] = None
    position: Optional[str] = None  # "GK", "DEF", "MID", "FWD"
    importance: float = 0.0  # 0-1, derived from xG or minutes played


@dataclass
class TeamInjuries:
    team: str
    out: list[PlayerStatus] = field(default_factory=list)
    doubtful: list[PlayerStatus] = field(default_factory=list)
    source: str = "manual"
    last_updated: str = ""

    def attack_penalty(self) -> float:
        """Penalización al ataque (0-1). 0 = sin lesionados, 0.5 = mucho."""
        fwd_out = [p for p in self.out if p.position in ("FWD", "MID")]
        total = sum(p.importance for p in fwd_out)
        return min(0.5, total * 0.4)

    def defense_penalty(self) -> float:
        """Penalización a la defensa (0-1)."""
        def_out = [p for p in self.out if p.position in ("DEF", "GK")]
        total = sum(p.importance for p in def_out)
        return min(0.3, total * 0.3)


def load_injuries(path: Path = INJURIES_PATH) -> dict[str, TeamInjuries]:
    """Carga el archivo de lesionados. Si no existe, retorna dict vacío."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for team, info in data.items():
        ti = TeamInjuries(
            team=team,
            source=info.get("source", "manual"),
            last_updated=info.get("last_updated", ""),
        )
        for p in info.get("out", []):
            ti.out.append(PlayerStatus(**p))
        for p in info.get("doubtful", []):
            ti.doubtful.append(PlayerStatus(**p))
        result[team] = ti
    return result


def save_injuries(injuries: dict[str, TeamInjuries], path: Path = INJURIES_PATH) -> None:
    """Guarda el archivo de lesionados."""
    data = {}
    for team, ti in injuries.items():
        data[team] = {
            "source": ti.source,
            "last_updated": ti.last_updated,
            "out": [
                {
                    "name": p.name,
                    "reason": p.reason,
                    "expected_return": p.expected_return,
                    "position": p.position,
                    "importance": p.importance,
                }
                for p in ti.out
            ],
            "doubtful": [
                {
                    "name": p.name,
                    "reason": p.reason,
                    "expected_return": p.expected_return,
                    "position": p.position,
                    "importance": p.importance,
                }
                for p in ti.doubtful
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
