"""Scraper de Wikipedia + RSS para injury/roster news.

Wikipedia: current squad info, coaches, recent form
RSS: injury news from feeds (BBC, ESPN, etc)

Rate limiting:
- Wikipedia: 1 req/seg (politica oficial de Wikimedia)
- RSS: 1 req/feed (manual call, no auto-poll)

Uso:
    from src.data.scraping import get_team_roster, get_injury_news

    roster = get_team_roster("Argentina")  # Wikipedia
    news = get_injury_news("Argentina")      # RSS
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.paths import PROCESSED_DIR

# Configuracion
USER_AGENT = "predictor-mundial/1.0 (https://github.com/PedroSL0904/predictor-mundial)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_MIN_INTERVAL = 1.1  # segundos entre requests (rate limit Wikimedia)
RSS_MIN_INTERVAL = 2.0  # segundos entre RSS feeds

# Default RSS feeds (injury/soccer news)
DEFAULT_RSS_FEEDS = [
    "https://www.espn.com/espn/rss/soccer/news",
    "http://feeds.bbci.co.uk/sport/football/rss.xml",
]

# Estado global de rate limiting
_last_request_time: dict[str, float] = {}


def rate_limited(min_interval: float) -> Callable:
    """Decorator que enforce un min_interval entre llamadas."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = func.__name__
            now = time.time()
            last = _last_request_time.get(key, 0.0)
            elapsed = now - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            _last_request_time[key] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


@dataclass
class RosterEntry:
    """Una entrada del roster de un equipo (jugador)."""

    name: str
    position: str = ""
    number: int | None = None
    club: str = ""
    age: int | None = None


@dataclass
class TeamRoster:
    """Roster + metadata de un equipo desde Wikipedia."""

    team: str
    url: str = ""
    coach: str = ""
    captain: str = ""
    players: list[RosterEntry] = field(default_factory=list)
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "url": self.url,
            "coach": self.coach,
            "captain": self.captain,
            "players": [
                {
                    "name": p.name,
                    "position": p.position,
                    "number": p.number,
                    "club": p.club,
                    "age": p.age,
                }
                for p in self.players
            ],
            "fetched_at": self.fetched_at,
        }


@rate_limited(WIKIPEDIA_MIN_INTERVAL)
def _wikipedia_search_team(team: str) -> str | None:
    """Busca el titulo de Wikipedia para el equipo. Retorna page title o None."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{team} national football team",
        "format": "json",
        "srlimit": 1,
    }
    try:
        r = requests.get(
            WIKIPEDIA_API, params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except Exception as e:
        print(f"  Wikipedia search error for {team}: {e}")
    return None


@rate_limited(WIKIPEDIA_MIN_INTERVAL)
def _wikipedia_get_page(title: str) -> str:
    """Obtiene el HTML de una pagina de Wikipedia."""
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": 1,
    }
    r = requests.get(
        WIKIPEDIA_API, params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("parse", {}).get("text", {}).get("*", "")


def _parse_roster_html(html: str, team: str) -> TeamRoster:
    """Parsea el HTML de Wikipedia para extraer roster + coach + captain."""
    soup = BeautifulSoup(html, "html.parser")
    roster = TeamRoster(team=team, fetched_at=datetime.now(UTC).isoformat())

    # URL
    canonical = soup.find("link", rel="canonical")
    if canonical:
        roster.url = canonical.get("href", "")

    # Coach y captain: busqueda por infobox
    infobox = soup.find("table", class_="infobox")
    if infobox:
        for row in infobox.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td:
                continue
            label = th.get_text(strip=True).lower()
            value = td.get_text(strip=True)
            if "head coach" in label or "manager" in label:
                roster.coach = value
            elif "captain" in label:
                roster.captain = value

    # Roster: tablas con clase 'wikitable'
    tables = soup.find_all("table", class_="wikitable")
    for tbl in tables:
        rows = tbl.find_all("tr")
        if not rows:
            continue
        # Chequear si parece una tabla de squad (header con 'Player' o 'Name')
        header_cells = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        if not any(h in header_cells for h in ["player", "name", "no.", "pos."]):
            continue
        # Detectar columnas
        col_map = {}
        for i, h in enumerate(header_cells):
            if h in ("no.", "no", "#"):
                col_map["number"] = i
            elif h in ("player", "name"):
                col_map["name"] = i
            elif h in ("position", "pos.", "pos"):
                col_map["position"] = i
            elif h in ("club", "current club"):
                col_map["club"] = i
            elif h == "age":
                col_map["age"] = i
        # Si no hay columna name, no podemos hacer nada
        if "name" not in col_map:
            continue
        # Parsear filas
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            try:
                name_cell_idx = col_map["name"]
                if name_cell_idx >= len(cells):
                    continue
                name = cells[name_cell_idx].get_text(strip=True)
                if not name or len(name) < 2:
                    continue
                # Quitar cosas como "[1]" de referencias
                name = name.split("[")[0].strip()
                number = None
                if "number" in col_map and col_map["number"] < len(cells):
                    try:
                        number = int(cells[col_map["number"]].get_text(strip=True))
                    except (ValueError, IndexError):
                        pass
                position = cells[col_map["position"]].get_text(strip=True) if "position" in col_map and col_map["position"] < len(cells) else ""
                club = cells[col_map["club"]].get_text(strip=True) if "club" in col_map and col_map["club"] < len(cells) else ""
                age = None
                if "age" in col_map and col_map["age"] < len(cells):
                    try:
                        age = int(cells[col_map["age"]].get_text(strip=True))
                    except (ValueError, IndexError):
                        pass
                roster.players.append(RosterEntry(
                    name=name, position=position, number=number,
                    club=club, age=age,
                ))
            except (IndexError, AttributeError):
                continue
        if roster.players:
            break  # Solo la primera tabla de squad

    return roster


def get_team_roster(team: str) -> TeamRoster:
    """Obtiene el roster actual de un equipo desde Wikipedia.

    Returns:
        TeamRoster (puede tener listas vacias si no se encuentra).
    """
    title = _wikipedia_search_team(team)
    if not title:
        return TeamRoster(team=team, fetched_at=datetime.utcnow().isoformat())
    html = _wikipedia_get_page(title)
    return _parse_roster_html(html, team)


@dataclass
class NewsItem:
    """Item de noticia de RSS."""

    title: str
    link: str
    description: str
    published: str
    source: str

    def mentions_team(self, team: str) -> bool:
        """Chequea si el item menciona al equipo."""
        text = f"{self.title} {self.description}".lower()
        return team.lower() in text


@rate_limited(RSS_MIN_INTERVAL)
def _fetch_rss(url: str) -> list[NewsItem]:
    """Fetch y parse un RSS feed."""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  RSS fetch error for {url}: {e}")
        return []
    items: list[NewsItem] = []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"  RSS parse error for {url}: {e}")
        return []
    # Soportar RSS 2.0 (channel/item) y Atom (feed/entry)
    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            desc = item.findtext("description", "")
            pub = item.findtext("pubDate", "")
            items.append(NewsItem(
                title=title, link=link, description=desc,
                published=pub, source=url,
            ))
    else:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns)
            link_el = entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            desc = entry.findtext("atom:summary", "", ns) or entry.findtext("atom:content", "", ns)
            pub = entry.findtext("atom:updated", "", ns)
            items.append(NewsItem(
                title=title, link=link, description=desc or "",
                published=pub, source=url,
            ))
    return items


def get_injury_news(
    team: str,
    feeds: list[str] | None = None,
) -> list[NewsItem]:
    """Busca noticias que mencionen al equipo en los feeds RSS.

    Args:
        team: nombre del equipo a buscar.
        feeds: lista de URLs de feeds RSS. Si None, usa DEFAULT_RSS_FEEDS.

    Returns:
        Lista de NewsItem que mencionan al equipo.
    """
    if feeds is None:
        feeds = DEFAULT_RSS_FEEDS
    matches: list[NewsItem] = []
    for feed in feeds:
        items = _fetch_rss(feed)
        for item in items:
            if item.mentions_team(team):
                matches.append(item)
    return matches


def save_roster_to_cache(roster: TeamRoster, cache_dir: Path = PROCESSED_DIR) -> Path:
    """Guarda el roster en cache JSON."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in roster.team.lower())
    path = cache_dir / f"roster_{safe_name}.json"
    import json
    with open(path, "w") as f:
        json.dump(roster.to_dict(), f, indent=2)
    return path
