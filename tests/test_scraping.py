"""Tests del modulo de scraping (Wikipedia + RSS).

Solo testea las funciones de transformacion. NO hace requests de red
(los tests de integracion con red se ejecutan manualmente).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from src.data.scraping import NewsItem, _parse_roster_html


def test_parse_roster_html_empty() -> None:
    """HTML vacio devuelve roster vacio."""
    roster = _parse_roster_html("", "Argentina")
    assert roster.team == "Argentina"
    assert roster.players == []
    assert roster.coach == ""


def test_parse_roster_html_basic() -> None:
    """HTML con tabla wikitabe basica se parsea."""
    html = """
    <html>
    <head>
      <link rel="canonical" href="https://en.wikipedia.org/wiki/Argentina_national_football_team" />
    </head>
    <body>
      <table class="infobox">
        <tr><th>Head coach</th><td>Lionel Scaloni</td></tr>
        <tr><th>Captain</th><td>Lionel Messi</td></tr>
      </table>
      <table class="wikitable">
        <tr><th>No.</th><th>Pos.</th><th>Player</th><th>Club</th><th>Age</th></tr>
        <tr><td>1</td><td>GK</td><td>Emiliano Martínez</td><td>Aston Villa</td><td>32</td></tr>
        <tr><td>10</td><td>FW</td><td>Lionel Messi</td><td>Inter Miami</td><td>37</td></tr>
      </table>
    </body>
    </html>
    """
    roster = _parse_roster_html(html, "Argentina")
    assert roster.coach == "Lionel Scaloni"
    assert roster.captain == "Lionel Messi"
    assert roster.url == "https://en.wikipedia.org/wiki/Argentina_national_football_team"
    assert len(roster.players) == 2
    assert roster.players[0].name == "Emiliano Martínez"
    assert roster.players[0].position == "GK"
    assert roster.players[0].number == 1
    assert roster.players[0].age == 32
    assert roster.players[1].name == "Lionel Messi"
    assert roster.players[1].number == 10


def test_parse_roster_html_strips_references() -> None:
    """Las referencias [1] en nombres se quitan."""
    html = """
    <html><body>
    <table class="wikitable">
      <tr><th>Player</th></tr>
      <tr><td>Cristiano Ronaldo[1]</td></tr>
    </table>
    </body></html>
    """
    roster = _parse_roster_html(html, "Portugal")
    assert len(roster.players) == 1
    assert roster.players[0].name == "Cristiano Ronaldo"


def test_parse_roster_html_no_squad_table() -> None:
    """Si no hay tabla de squad, devuelve players=[]."""
    html = """
    <html><body>
    <table class="wikitable">
      <tr><th>Year</th><th>Result</th></tr>
      <tr><td>2018</td><td>Round of 16</td></tr>
    </table>
    </body></html>
    """
    roster = _parse_roster_html(html, "Argentina")
    assert roster.players == []


def test_news_item_mentions_team() -> None:
    item = NewsItem(
        title="Argentina wins World Cup",
        link="https://example.com",
        description="Argentina beat France in the final",
        published="2022-12-18",
        source="https://espn.com",
    )
    assert item.mentions_team("Argentina") is True
    assert item.mentions_team("France") is True
    assert item.mentions_team("Brazil") is False


def test_news_item_mentions_case_insensitive() -> None:
    item = NewsItem(
        title="ARGENTINA wins",
        link="",
        description="",
        published="",
        source="",
    )
    assert item.mentions_team("Argentina") is True
    assert item.mentions_team("argentina") is True


def test_rss_parse_rss2() -> None:
    """Parse RSS 2.0 con channel/item."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Feed</title>
        <item>
          <title>Argentina news</title>
          <link>https://example.com/1</link>
          <description>Argentina won</description>
          <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
        </item>
        <item>
          <title>Other news</title>
          <link>https://example.com/2</link>
          <description>Nothing about Argentina</description>
          <pubDate>Mon, 02 Jan 2024 12:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """
    root = ET.fromstring(xml)
    channel = root.find("channel")
    items = []
    for item in channel.findall("item"):
        items.append(NewsItem(
            title=item.findtext("title", ""),
            link=item.findtext("link", ""),
            description=item.findtext("description", ""),
            published=item.findtext("pubDate", ""),
            source="test",
        ))
    assert len(items) == 2
    assert items[0].title == "Argentina news"
    assert items[0].published == "Mon, 01 Jan 2024 12:00:00 GMT"


def test_rss_parse_atom() -> None:
    """Parse Atom con feed/entry."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Test Atom</title>
      <entry>
        <title>Brazil wins</title>
        <link href="https://example.com/atom/1"/>
        <summary>Brazil won</summary>
        <updated>2024-01-01T12:00:00Z</updated>
      </entry>
    </feed>
    """
    root = ET.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", "", ns)
        link_el = entry.find("atom:link", ns)
        link = link_el.get("href", "") if link_el is not None else ""
        items.append(NewsItem(
            title=title, link=link, description="",
            published=entry.findtext("atom:updated", "", ns),
            source="test-atom",
        ))
    assert len(items) == 1
    assert items[0].title == "Brazil wins"
    assert items[0].link == "https://example.com/atom/1"
