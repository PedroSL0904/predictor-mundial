# Predictor Mundial 2026

Sistema de predicciones de fútbol basado en Poisson/Dixon-Coles con ponderación por Elo rolling y xG histórico.

## Objetivo

Construir el mejor modelo predictivo open-source para partidos de fútbol de selecciones, optimizado para Brier score y RPS sobre Mundiales 2014, 2018 y 2022.

## Rendimiento vs. los mejores sistemas del mundo

**Pregunta honesta**: ¿qué tan bien estamos comparados con sistemas profesionales como Pinnacle, Opta, FiveThirtyEight?

| Sistema | Brier 1X2 | Sign accuracy | Fuente |
|---|---|---|---|
| **Pinnacle closing odds** (mercado) | **0.20-0.22** | 53-55% | Mercado real (mejor predictor) |
| Opta / Stats Perform | 0.21-0.23 | 53-55% | Industria |
| FiveThirtyEight SPI (2014-2022) | 0.22-0.24 | 52-54% | Modelo estadístico |
| **Predictor-Mundial** (backtest 2014-2022) | **0.578** | **54.0%** | Mi sistema |
| **Predictor-Mundial** (WC 2026 in-progress, 34 partidos) | **0.569** | **58.8%** | Mi sistema |
| Baseline uniforme (1/3) | 0.667 | 33% | Random |

**Lectura**:

1. **Estamos en el rango de sign accuracy de Pinnacle** (54% vs 53-55%): acierto de signo comparable al mejor mercado del mundo.

2. **Pero estamos ~0.35 por encima en Brier**: nuestras probabilidades están peor calibradas. Una predicción de "70% H" debería acertar 70%, y la nuestra no tanto.

3. **¿Por qué Pinnacle es mucho mejor en Brier?**

   - **Información de mercado en tiempo real**: lesionados de último momento, clima, alineaciones titulares.
   - **Sharp bettors corrigen el precio**: el mercado agrega información de miles de apostadores profesionales (los modelos "buenos" son los que Platense Inc. y demás sharp bettors usan).
   - **Modelos más complejos**: xG player-level, player ratings, fatiga, etc.
   - **Datos propietarios**: Opta tiene tracking de eventos de cada partido (pases, tiros, pressing, etc.).
   - **Capacidad de mover la línea**: Pinnacle cierra cerca del "true" probability porque acepta límites altos y puede mover líneas.

4. **WC 2026 va mejor que el backtest histórico** (Brier 0.569 vs 0.578). Esto puede ser:
   - Suerte positiva en 34 partidos
   - El WC 2026 tiene más variabilidad (más empates, más sorpresas)
   - El ajuste `league_avg_multiplier=1.18` ayudó

5. **Conclusión**: somos un **modelo amateur competitivo** comparable en sign accuracy al mercado, pero sin su información en tiempo real. Para cerrar la brecha en Brier (~0.35), necesitaríamos:
   - Datos de lesionados en tiempo real
   - Cuotas de mercado en tiempo real como feature
   - Modelo de xG player-level
   - Ensemble con Pinnacle/Opta

### Comparación con Oloraculo (sistema de referencia)

| Sistema | Brier | Sign accuracy | Aciertos |
|---|---|---|---|
| Oloraculo (WC 2026, 23 partidos) | 0.614 | 52.2% | 12/23 |
| **Predictor-Mundial** (34 partidos) | **0.569** | **58.8%** | ~20/34 |

Ganamos en Brier (-0.045) y sign (+7pp). Las mejoras vienen de recent form, draw_boost, y elo_gap_inflation que Oloraculo no usa.

## Predicciones del Mundial 2026

Ver [WC2026_README.md](WC2026_README.md) para las predicciones actualizadas partido a partido (se re-generan ejecutando `python src/cli/wc2026_readme.py`).

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
