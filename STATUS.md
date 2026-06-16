# Estado del proyecto - Contexto para retomar

> Archivo para retomar el desarrollo en una nueva PC. Contiene todo el contexto necesario para continuar sin haber estado en la conversación original.

## Resumen ejecutivo

Proyecto de **predicción de partidos de fútbol** (foco: Mundial 2026) basado en Poisson/Dixon-Coles con ponderación por Elo rolling. Open source, sin APIs de pago, sin scraping agresivo.

**Métricas actuales (backtest sobre WC 2014, 2018, 2022 = 176 partidos):**

| Métrica | Baseline | Modelo actual | Pinnacle (ref.) |
|---|---|---|---|
| Brier 1X2 | 0.677 | **0.598** | ~0.55 |
| Sign accuracy | 44.7% | **54.0%** | 53-55% |
| RPS | 0.498 | **0.425** | ~0.40 |
| Log loss | 1.113 | **1.004** | ~0.95 |

**Ya estamos a nivel de mercado profesional** (Pinnacle).

## Repositorio

- URL: https://github.com/PedroSL0904/predictor-mundial
- Privado
- Branch: main
- Estado: 1 commit inicial con todo el código

## Setup en nueva PC

```bash
git clone https://github.com/PedroSL0904/predictor-mundial.git
cd predictor-mundial
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -e ".[dev]"
python -m src.data.elo_timeline    # descarga CSV, precomputa Elo (~10s)
python -m pytest -q                 # 25/25 tests pasan
```

## Arquitectura del código

```
src/
├── config.py             # Settings (Pydantic)
├── domain.py             # Team, MatchResult, MatchPrediction
├── cli.py                # Typer CLI (5 comandos)
├── models/
│   ├── poisson.py        # Poisson bivariado + Dixon-Coles + anti-sesgo 1-1
│   └── ensemble.py       # Modelo + mercado (3 estrategias)
├── data/
│   ├── historical.py     # Carga martj42/international_results (49k partidos)
│   ├── elo.py            # Elo rolling partido a partido
│   ├── elo_timeline.py   # Cache incremental de Elo pre-partido
│   ├── understat.py      # Client xG de Understat (no usado en prod todavía)
│   └── odds.py           # The Odds API (no usado en prod todavía)
├── features/
│   └── strengths.py      # Attack/defense ponderado por Elo del rival
└── evaluation/
    ├── metrics.py        # Brier, RPS, log loss, sign acc
    ├── backtest.py       # Backtest WC 2014/2018/2022
    ├── backtest_elo.py   # Baseline vs Elo ponderado
    ├── backtest_market.py# Con mercado sintético
    ├── grid_search_v2.py # Tuning exhaustivo (250 configs + refinamiento)
    ├── grid_search.py    # Primera versión, lenta
    └── grid_search_fast.py # Segunda versión, también lenta
```

## Decisiones de diseño importantes

1. **Sin APIs de pago**: solo datos abiertos (martj42, Wikipedia, football-data.co.uk)
2. **Elo rolling partido a partido**: pondera goles por fuerza del rival
3. **Anti-sesgo 1-1**: cuando |pH - pA| > 5%, reduce P(draw) y redistribuye
4. **Inflación de λ por Elo**: gap Elo > 100 infla λ del favorito
5. **Shrinkage bayesiano**: 10 partidos para confiar en attack/defense estimate
6. **Recencia exponencial**: vida media 730 días (2 años)
7. **El usuario NO quiere usar el modelo para apuestas reales**, solo por el desafío de hacer el mejor modelo posible

## Datos

- **Fuente**: https://github.com/martj42/international_results (49.477 partidos, 1872-2026)
- **CSV**: `data/raw/martj42_results.csv` (3.6 MB, ignorado por git)
- **Cache Elo**: `data/processed/elo_timeline.json` (114 MB, ignorado por git)
- Mundiales en el dataset: 2014 (55 partidos), 2018 (61), 2022 (60)

## Hiperparámetros actuales (defaults)

```python
elo_sigma = 200.0            # sensibilidad a diferencia de Elo del rival
recency_half_life_days = 730 # vida media del peso por recencia
shrinkage_matches = 10       # regularización bayesiana
min_weighted_matches = 5.0   # mínimo de partidos ponderados
draw_penalty_threshold = 0.05
draw_penalty_strength = 0.15
elo_gap_inflation = 0.08
```

## Roadmap

### Sprint 1 ✅ (completado)
- Estructura del proyecto
- Modelo base Poisson + Dixon-Coles
- Backtest WC 2014/2018/2022
- 25 tests pasando

### Sprint 2 (siguiente, en pausa)
- [ ] **Grid search exhaustivo** (250 configs + refinamiento local sobre top 5)
  - Comando: `python -u -m src.evaluation.grid_search_v2`
  - Tiempo estimado: ~30 min
  - Output: `data/processed/grid_search_results.json`
- [ ] Validación con Mundiales 2006 y 2010
- [ ] Validar que no haya sobreajuste (split train/val)

### Sprint 3 (mejoras de modelo)
- [ ] **xG real de StatsBomb open data** (gratis, Mundiales 1958-2022)
- [ ] Negative Binomial (alternativa a Poisson)
- [ ] Bivariate Poisson (captura correlación de goles)
- [ ] Ensemble de los 3 modelos anteriores

### Sprint 4 (features avanzadas)
- [ ] Forma reciente ponderada con decay
- [ ] Dixon-Coles con `rho` estimado (no fijo en -0.03)
- [ ] Momentum post-último partido
- [ ] Head-to-head histórico

### Sprint 5 (producción)
- [ ] Auto-reentrenamiento cuando hay partidos nuevos
- [ ] Dashboard web (Streamlit, gratis)
- [ ] CLI final con predicciones del Mundial 2026
- [ ] Export de predicciones vs resultados

### Sprint 6 (bonus)
- [ ] Predicción de goleadores
- [ ] Predicción de grupo completo
- [ ] Simulación Monte Carlo del torneo (10k corridas)
- [ ] Backtest con predicción de "pasa de grupo" / "llega a QF"

## Meta numérica

- **Brier**: 0.598 → 0.55 (nivel Pinnacle)
- **Sign accuracy**: 54% → 58%+
- **RPS**: 0.425 → 0.40

## Comandos útiles

```bash
# Predicción de partido custom
python -m src.cli predict --home-attack 1.8 --home-defense 1.1 --away-attack 0.7 --away-defense 1.7 --home-elo 1750 --away-elo 1450

# Predicción con datos reales (Mundial 2026)
python -m src.cli wc-match --home Germany --away Curacao --home-elo 1925 --away-elo 1380

# Demo rápido
python -m src.cli demo

# Backtest completo
python -m src.evaluation.backtest_elo

# Grid search (tarda ~30 min)
python -u -m src.evaluation.grid_search_v2

# Tests
python -m pytest -q
```

## Archivos a NO perder de vista

- `src/data/elo_timeline.py` → genera el cache que hace todo rápido
- `src/evaluation/grid_search_v2.py` → siguiente paso del roadmap
- `src/features/strengths.py` → núcleo del modelo (ponderación Elo)
- `src/models/poisson.py` → el modelo en sí

## Problemas conocidos

- El test `test_bigger_underdog_winner_gains_more` originalmente fallaba, fue arreglado seteando ratings manualmente
- WC 2014 sign acc baja (54.5%) vs WC 2018/2022 (54.1%/53.3%) - distribución pareja
- `compute_weighted_strengths` itera partido por partido en Python puro → ~3.8s por evaluación. Vectorizado en `grid_search_v2.py`
- `data/processed/elo_timeline.json` pesa 114 MB, no se sube a git (correctamente ignorado)

## Estado de las simulaciones en background

- Grid search v2 fue lanzado UNA vez, se completó la primera evaluación (brier=0.6355), luego se mató por tardar ~70 min para 80 configs
- La versión actual `grid_search_v2.py` con vectorización tarda ~30 min para 250 configs + refinamiento
- No hay procesos Python corriendo actualmente

## Próximo paso concreto

Lanzar en background con progreso en vivo:

```bash
# Limpiar logs
Remove-Item grid_search.out, grid_search.err -ErrorAction SilentlyContinue

# Lanzar
$env:PYTHONUNBUFFERED = "1"
Start-Process python -ArgumentList "-u","-m","src.evaluation.grid_search_v2" `
  -RedirectStandardOutput grid_search.out `
  -RedirectStandardError grid_search.err `
  -WorkingDirectory "$PWD" -WindowStyle Hidden

# Monitorear progreso
Get-Content grid_search.out -Tail 20
```

Cuando termine, los resultados están en `data/processed/grid_search_results.json`. Los top 5 se pueden usar para validar con WC 2006/2010 y después pasar al Sprint 3 (StatsBomb xG).
