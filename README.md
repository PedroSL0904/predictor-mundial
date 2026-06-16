# Predictor Mundial 2026

Sistema de predicciones de fútbol basado en Poisson/Dixon-Coles con ponderación por Elo rolling y xG histórico.

## Objetivo

Construir el mejor modelo predictivo open-source para partidos de fútbol de selecciones, optimizado para Brier score y RPS sobre Mundiales 2014, 2018 y 2022.

## Estado actual

| Métrica | Baseline | Modelo actual | Pinnacle (ref.) |
|---|---|---|---|
| Brier 1X2 | 0.677 | **0.598** | ~0.55 |
| Sign accuracy | 44.7% | **54.0%** | 53-55% |
| RPS | 0.498 | **0.425** | ~0.40 |

## Arquitectura

```
predictor-mundial/
├── src/
│   ├── config.py             # Settings (Pydantic)
│   ├── domain.py             # Team, MatchResult, MatchPrediction
│   ├── cli.py                # Typer CLI
│   ├── models/
│   │   ├── poisson.py        # Poisson bivariado + Dixon-Coles + anti-sesgo 1-1
│   │   └── ensemble.py       # Modelo + mercado
│   ├── data/
│   │   ├── historical.py     # CSV de martj42/international_results
│   │   ├── elo.py            # Elo rolling partido a partido
│   │   ├── elo_timeline.py   # Cache incremental de Elo pre-partido
│   │   ├── understat.py      # Client xG de Understat
│   │   └── odds.py           # The Odds API (Pinnacle)
│   ├── features/
│   │   └── strengths.py      # Attack/defense ponderado por Elo
│   └── evaluation/
│       ├── metrics.py        # Brier, RPS, log loss
│       ├── backtest.py       # Backtest WC 2014/2018/2022
│       ├── backtest_elo.py   # Baseline vs Elo ponderado
│       ├── backtest_market.py# Con mercado sintético
│       └── grid_search_v2.py # Tuning hiperparámetros
├── tests/
│   ├── test_poisson.py       # 13 tests del modelo
│   └── test_elo.py           # 12 tests del Elo
├── data/
│   ├── raw/                  # CSVs descargados (gitignored)
│   └── processed/            # Cache de features (gitignored)
└── pyproject.toml
```

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -e ".[dev]"
```

## Uso

### Precomputar timeline de Elo (~10s)
```bash
python -m src.data.elo_timeline
```

### Predecir un partido del Mundial 2026
```bash
python -m src.cli wc-match --home Germany --away Curacao --home-elo 1925 --away-elo 1380
```

### Correr backtest
```bash
python -m src.evaluation.backtest_elo       # ~2 min
python -u -m src.evaluation.grid_search_v2  # ~30 min, 250 configs
```

### Tests
```bash
pytest -q
```

## Decisiones de diseño

- **Sin APIs pagas**: solo datos abiertos (martj42, Wikipedia, football-data.co.uk)
- **Elo rolling partido a partido**: pondera los goles según fuerza del rival
- **Anti-sesgo 1-1**: penaliza empates cuando hay gap Elo grande
- **Inflación de λ por Elo**: goleadas extremas se modelan mejor

## Próximos pasos

- [ ] Grid search exhaustivo de hiperparámetros
- [ ] xG real desde StatsBomb open data
- [ ] Negative Binomial / Bivariate Poisson
- [ ] Validación con Mundiales 2006, 2010
- [ ] Dashboard web con auto-reentrenamiento
