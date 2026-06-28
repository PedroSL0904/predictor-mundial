"""CLI: actualizar injuries.json usando OpenCode LLM (minimax-m3).

Uso:
    export OPENCODE_API_KEY=sk-...
    python -m src.cli.update_injuries

Hace una sola llamada al LLM con los 32 equipos calificados a R32 y
parsea la respuesta como JSON para actualizar data/processed/injuries.json.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date

from src.data.injuries import (
    INJURIES_PATH,
    PlayerStatus,
    TeamInjuries,
    load_injuries,
    save_injuries,
)
from src.data.team_names import OLO_TO_MARTJ
from src.data.wc2026_fixture import generate_group_fixtures
from src.llm.client import OpenCodeClient
from src.logging_config import get_logger
from src.simulation.r32_predictions import (
    compute_group_standings,
    get_top_8_thirds,
)

logger = get_logger(__name__)

# Mapeo inverso: MARTJ -> OLO (usado en el fixture)
MARTJ_TO_OLO = {v: k for k, v in OLO_TO_MARTJ.items()}


ENRICH_SYSTEM = """Eres un asistente que valida y enriquece datos de
planteles de futbol. Recibes una lista de jugadores que se reportan
como lesionados para una seleccion. Tu trabajo:
1. Verifica que los jugadores son reales (no corrijas errores de nombre
   menores, solo si son completamente incorrectos).
2. Para CADA jugador de la lista, asigna position (GK/DEF/MID/FWD)
   e importance (0-1, donde 0.7-1.0=titular, 0.3-0.6=suplente habitual,
   0.1-0.3=rotacion).
3. NO agregues jugadores nuevos. NO omitas jugadores.
4. Responde SOLO con JSON valido, sin markdown."""


DISCOVERY_SYSTEM = """Eres un asistente que identifica jugadores
lesionados en planteles del Mundial 2026. Tu conocimiento llega hasta
enero 2026. SOLO reportas jugadores que estaban confirmados como
lesionados/suspendidos ANTES del torneo segun noticias publicas.
NO inventes. Si no sabes, devuelve listas vacias.
ESQUEMA: {"out": [{"name": str, "position": GK/DEF/MID/FWD,
"importance": 0-1}], "doubtful": [...]}. importance 0.7-1.0 titular,
0.3-0.6 suplente habitual, 0.1-0.3 rotacion."""


DISCOVERY_PROMPT = """De tu conocimiento pre-torneo (enero 2026 o antes),
lista jugadores lesionados o en duda para el mundial de estos equipos:

{teams}

Responde SOLO con JSON asi (sin markdown):
{{"Argentina": {{"out": [...], "doubtful": [...]}}, ...}}
Si no sabes de un equipo: {{"out": [], "doubtful": []}}."""


ENRICH_PROMPT_TEMPLATE = """Para {team}, se reportan estos jugadores
lesionados o en duda para el Mundial 2026:

OUT (no jugaran):
{out_list}

DOUBTFUL (50% prob de jugar):
{doubt_list}

Para CADA jugador de las listas de arriba, completa position
(GK/DEF/MID/FWD) y importance (0-1).

Responde SOLO con JSON (sin markdown):
{{"out": [{{"name": "...", "position": "DEF", "importance": 0.6}}],
"doubtful": [...]}}"""


def discover_injuries(client: OpenCodeClient, qualified: list[str]) -> dict:
    """Pregunta al LLM sobre lesionados de los 32 equipos (1 llamada)."""
    teams_str = "\n".join(f"- {t}" for t in qualified)
    prompt = DISCOVERY_PROMPT.format(teams=teams_str)
    logger.info("  Llamada discovery: 32 equipos...", end=" ")
    t0 = time.time()
    response = client.messages(
        prompt=prompt,
        system=DISCOVERY_SYSTEM,
        max_tokens=4000,
        temperature=0.0,
    )
    logger.info(f"{time.time() - t0:.1f}s ({response.input_tokens}+{response.output_tokens} tokens)")
    try:
        return extract_json(response.text)
    except Exception as e:
        logger.info(f"  ERROR discovery: {e}")
        return {}


def enrich_team_injuries(
    client: OpenCodeClient, team: str, ti: TeamInjuries,
) -> TeamInjuries:
    """Para un equipo con jugadores, pregunta al LLM por position/importance."""
    if not ti.out and not ti.doubtful:
        return ti

    out_list = "\n".join(f"- {p.name}" for p in ti.out) or "(ninguno)"
    doubt_list = "\n".join(f"- {p.name}" for p in ti.doubtful) or "(ninguno)"
    prompt = ENRICH_PROMPT_TEMPLATE.format(
        team=team, out_list=out_list, doubt_list=doubt_list,
    )
    try:
        response = client.messages(
            prompt=prompt, system=ENRICH_SYSTEM,
            max_tokens=500, temperature=0.0,
        )
        data = extract_json(response.text)
    except Exception as e:
        logger.info(f"  ERROR enrich {team}: {e}")
        return ti

    # Mezclar resultados: conservar jugadores que el LLM confirma,
    # agregar position/importance del LLM
    confirmed_out = {p["name"].lower(): p for p in data.get("out", [])}
    confirmed_doubt = {p["name"].lower(): p for p in data.get("doubtful", [])}

    for p in ti.out:
        llm = confirmed_out.get(p.name.lower())
        if llm:
            if llm.get("position"):
                p.position = llm["position"]
            if llm.get("importance") is not None:
                p.importance = float(llm["importance"])
    for p in ti.doubtful:
        llm = confirmed_doubt.get(p.name.lower())
        if llm:
            if llm.get("position"):
                p.position = llm["position"]
            if llm.get("importance") is not None:
                p.importance = float(llm["importance"])
    return ti


def extract_json(text: str) -> dict:
    """Extrae JSON de la respuesta del LLM, tolerando markdown."""
    # Buscar bloque JSON dentro de markdown
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    return json.loads(match.group(0))


def main() -> None:
    fixtures = generate_group_fixtures()
    standings = compute_group_standings(fixtures)
    thirds = get_top_8_thirds(standings)

    # qualified viene del fixture (OLO names, ej "USA")
    qualified_olo = []
    for _g, t in standings.items():
        qualified_olo.append(t[0][0])  # winner
        qualified_olo.append(t[1][0])  # runner up
    for _g, team, *_ in thirds:
        qualified_olo.append(team)

    # Convertir a MARTJ names (usados en injuries.json) ej "USA" -> "United States"
    qualified = sorted(set(OLO_TO_MARTJ.get(t, t) for t in qualified_olo))
    logger.info(f"Equipos calificados a R32: {len(qualified)}")

    client = OpenCodeClient()
    logger.info(f"\nUsando modelo {client.model}")
    data = discover_injuries(client, qualified)
    if not data:
        logger.info("Sin datos del LLM, abortando")
        return

    # Cargar datos manuales existentes (FUENTE DE VERDAD para listas de jugadores)
    manual = load_injuries()
    logger.info(f"  Datos manuales existentes: {len(manual)} equipos")

    # Construir injuries final
    injuries: dict[str, TeamInjuries] = {}

    # Normalizar nombres del LLM al formato del proyecto
    name_aliases = {
        "USA": "United States",
        "United States": "United States",
        "Iran": "IR Iran",
        "IR Iran": "IR Iran",
        "South Korea": "South Korea",
        "Korea Republic": "South Korea",
        "Ivory Coast": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Cape Verde": "Cape Verde",
        "Cabo Verde": "Cape Verde",
        "Curacao": "Curacao",
        "Curaçao": "Curacao",
        "Czech Republic": "Czechia",
        "Czechia": "Czechia",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "DR Congo": "Congo DR",
        "Congo DR": "Congo DR",
        "Democratic Republic of the Congo": "Congo DR",
    }

    def normalize_name(name: str) -> str:
        # Primero aliases, luego MARTJ->OLO (si LLM devolvio formato MARTJ)
        if name in name_aliases:
            return name_aliases[name]
        if name in MARTJ_TO_OLO:
            return MARTJ_TO_OLO[name]
        return name

    # Para cada equipo calificado, empezar con manual (si existe) o vacío
    for team in qualified:
        if team in manual and "manual" in manual[team].source.lower():
            # Copiar manual como base (FUENTE DE VERDAD)
            ti = TeamInjuries(
                team=team,
                source=manual[team].source,
                last_updated=str(date.today()),
            )
            for p in manual[team].out:
                ti.out.append(PlayerStatus(
                    name=p.name, reason=p.reason,
                    expected_return=p.expected_return,
                    position=p.position, importance=p.importance,
                ))
            for p in manual[team].doubtful:
                ti.doubtful.append(PlayerStatus(
                    name=p.name, reason=p.reason,
                    expected_return=p.expected_return,
                    position=p.position, importance=p.importance,
                ))
            injuries[team] = ti
        else:
            injuries[team] = TeamInjuries(
                team=team,
                source=f"opencode:{client.model}",
                last_updated=str(date.today()),
            )

    # Agregar jugadores del LLM que NO estan en manual
    for team, info in data.items():
        canonical = normalize_name(team)
        if canonical not in injuries:
            continue
        ti = injuries[canonical]
        existing_names_out = {p.name.lower() for p in ti.out}
        existing_names_doubt = {p.name.lower() for p in ti.doubtful}
        for p in info.get("out", []):
            name = p.get("name", "")
            if not name or name.lower() in existing_names_out:
                continue
            ti.out.append(PlayerStatus(
                name=name, reason="injury",
                position=p.get("position"),
                importance=float(p.get("importance", 0.5)),
            ))
        for p in info.get("doubtful", []):
            name = p.get("name", "")
            if not name or name.lower() in existing_names_doubt:
                continue
            ti.doubtful.append(PlayerStatus(
                name=name, reason="injury",
                position=p.get("position"),
                importance=float(p.get("importance", 0.5)),
            ))

    # Enriquecer position/importance con LLM para jugadores con datos faltantes
    logger.info("\nEnriqueciendo position/importance via LLM...")
    for team, ti in injuries.items():
        if ti.out or ti.doubtful:
            has_missing = any(
                not p.position or p.importance == 0.5  # default = missing
                for p in ti.out + ti.doubtful
            )
            if has_missing:
                injuries[team] = enrich_team_injuries(client, team, ti)
                logger.info(f"  {team}: {len(ti.out)} out, {len(ti.doubtful)} doubt")
            else:
                logger.info(f"  {team}: {len(ti.out)} out, {len(ti.doubtful)} doubt (completo)")

    logger.info(f"\nTotal equipos con datos: {len(injuries)}")
    n_out = sum(len(ti.out) for ti in injuries.values())
    n_doubt = sum(len(ti.doubtful) for ti in injuries.values())
    logger.info(f"  {n_out} jugadores out, {n_doubt} doubtful")

    save_injuries(injuries, INJURIES_PATH)
    logger.info(f"\nGuardado en {INJURIES_PATH}")


if __name__ == "__main__":
    main()
