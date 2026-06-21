"""CLI principal del predictor."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.config import get_settings
from src.data.historical import (
    compute_strengths_from_results,
    load_martj42_csv,
    normalize_team_name,
)
from src.models import PoissonGoalModel, TeamStrength

app = typer.Typer(help="Predictor de fútbol basado en xG + cuotas")
console = Console()


@app.command()
def predict(
    home_attack: float = typer.Option(..., help="xG/goles a favor por partido (local)"),
    home_defense: float = typer.Option(..., help="xG/goles en contra por partido (local)"),
    away_attack: float = typer.Option(..., help="xG/goles a favor por partido (visitante)"),
    away_defense: float = typer.Option(..., help="xG/goles en contra por partido (visitante)"),
    home_elo: float = typer.Option(1500.0),
    away_elo: float = typer.Option(1500.0),
    home_name: str = "Local",
    away_name: str = "Visitante",
) -> None:
    """Predice un partido dados los strengths de cada equipo."""
    model = PoissonGoalModel()
    home = TeamStrength(
        name=home_name, attack=home_attack, defense_vulnerability=home_defense
    )
    away = TeamStrength(
        name=away_name, attack=away_attack, defense_vulnerability=away_defense
    )
    pred = model.predict(home, away, home_elo, away_elo)

    table = Table(title=f"Prediccion: {home_name} vs {away_name}")
    table.add_column("Metrica", style="cyan")
    table.add_column("Valor", style="magenta")
    table.add_row("Modelo", pred.model)
    table.add_row("P(Home)", f"{pred.p_home:.1%}")
    table.add_row("P(Draw)", f"{pred.p_draw:.1%}")
    table.add_row("P(Away)", f"{pred.p_away:.1%}")
    table.add_row("lambda_home", f"{pred.lambda_home:.2f}")
    table.add_row("lambda_away", f"{pred.lambda_away:.2f}")
    table.add_row(
        "Marcador top",
        f"{pred.most_likely_score[0]}-{pred.most_likely_score[1]} "
        f"({pred.most_likely_score_prob:.1%})",
    )
    console.print(table)


@app.command()
def wc_match(
    home: str = typer.Option(..., help="Nombre del equipo local"),
    away: str = typer.Option(..., help="Nombre del equipo visitante"),
    home_elo: float = typer.Option(1500.0),
    away_elo: float = typer.Option(1500.0),
    data_path: Path = typer.Option(Path("data/raw/martj42_results.csv")),
    years_window: int = typer.Option(8),
    as_of: str = typer.Option("2026-06-15", help="Fecha de corte para histórico"),
) -> None:
    """Predice un partido del Mundial 2026 con datos reales."""
    if not data_path.exists():
        console.print(f"[red]No existe {data_path}. Ejecutá el download primero.[/red]")
        raise typer.Exit(1)

    df = load_martj42_csv(data_path)
    df = df[df["date"] < as_of]
    df = df[df["date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=365 * years_window)]

    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)

    strengths = compute_strengths_from_results(df, min_matches=5)
    h = strengths[strengths["team"] == home_norm]
    a = strengths[strengths["team"] == away_norm]

    if h.empty or a.empty:
        console.print(f"[red]No hay datos para {home_norm} o {away_norm}[/red]")
        raise typer.Exit(1)

    home_team = TeamStrength(
        name=home_norm,
        attack=float(h["attack"].iloc[0]),
        defense_vulnerability=float(h["defense_vulnerability"].iloc[0]),
        matches=int(h["matches"].iloc[0]),
    )
    away_team = TeamStrength(
        name=away_norm,
        attack=float(a["attack"].iloc[0]),
        defense_vulnerability=float(a["defense_vulnerability"].iloc[0]),
        matches=int(a["matches"].iloc[0]),
    )

    console.print(f"[bold]Usando {home_team.matches} partidos para {home_norm}, "
                  f"{away_team.matches} para {away_norm}[/bold]\n")

    predict(
        home_attack=home_team.attack,
        home_defense=home_team.defense_vulnerability,
        away_attack=away_team.attack,
        away_defense=away_team.defense_vulnerability,
        home_elo=home_elo,
        away_elo=away_elo,
        home_name=home_norm,
        away_name=away_norm,
    )


@app.command()
def backtest() -> None:
    """Corre el backtest sobre Mundiales 2014, 2018, 2022 (cached)."""
    from src.evaluation.backtest_cached import run_cached_comparison
    run_cached_comparison()


@app.command()
def demo() -> None:
    """Predice un partido de ejemplo (Germany vs Curacao estilo WC 2026)."""
    console.print("[bold]Demo: Germany vs Curacao (Mundial 2026)[/bold]\n")
    predict(
        home_attack=2.10,
        home_defense=0.85,
        away_attack=0.55,
        away_defense=1.95,
        home_elo=1925.0,
        away_elo=1380.0,
        home_name="Germany",
        away_name="Curacao",
    )


@app.command()
def config_show() -> None:
    """Muestra la configuracion cargada."""
    settings = get_settings()
    console.print(settings.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
