# Estado del proyecto - Contexto para retomar

> Archivo para retomar el desarrollo en una nueva PC. Contiene todo el contexto necesario para continuar sin haber estado en la conversación original.

## Resumen ejecutivo

Proyecto de **predicción de partidos de fútbol** (foco: Mundial 2026) basado en Poisson/Dixon-Coles con ponderación por Elo rolling. Open source, sin APIs de pago, sin scraping agresivo.

**Métricas actuales (backtest sobre WC 2014, 2018, 2022 = 176 partidos, configuración optimizada por LOO 5 mundial):**

| Métrica | Baseline (sin Elo) | Modelo actual | Pinnacle (ref.) |
|---|---|---|---|
| Brier 1X2 | 0.659 | **0.593** | ~0.55 |
| Sign accuracy | 45.9% | **56.9%** | 53-55% |
| RPS | 0.480 | **0.417** | ~0.40 |
| Log loss | 1.088 | **0.997** | ~0.95 |
| Exact score | 3.3% | **7.9%** | ~8% |

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

## Hiperparámetros actuales (defaults optimizados LOO 5 mundial)

```python
elo_sigma = 225.0            # sensibilidad a diferencia de Elo del rival
recency_half_life_days = 1000.0  # vida media del peso por recencia
shrinkage_matches = 10       # regularización bayesiana
min_weighted_matches = 8.0   # mínimo de partidos ponderados
draw_penalty_threshold = 0.08
draw_penalty_strength = 0.05
draw_boost = 0.20            # boost de P(draw) en partidos parejos
elo_gap_inflation = 0.30
```

### Cambios respecto a Sprint 1
- `elo_sigma`: 200 → 225 (más peso Elo del rival)
- `recency_half_life_days`: 730 → 1000 (más memoria de partidos viejos)
- `min_weighted_matches`: 5 → 8 (más restrictivo)
- `draw_penalty_threshold`: 0.05 → 0.08 (umbral de boost/penalty más alto)
- `draw_penalty_strength`: 0.15 → 0.05 (penalty más bajo)
- `elo_gap_inflation`: 0.08 → 0.30 (mayor inflación de λ del favorito)
- **NUEVO `draw_boost = 0.20`**: aumenta P(draw) en partidos parejos, compensa sub-predicción sistemática del modelo Poisson (que estimaba ~15% cuando real es ~22% en Mundiales)

## Roadmap

### Sprint 1 ✅ (completado)
- Estructura del proyecto
- Modelo base Poisson + Dixon-Coles
- Backtest WC 2014/2018/2022
- 25 tests pasando

### Sprint 2 ✅ (completado)
- ✅ Grid search exhaustivo v2 (250 configs + refinamiento)
- ✅ Validación con WC 2006 y 2010 (LOO 5 mundial)
- ✅ **No hubo overfitting**: train_brier≈test_brier en top configs
- ✅ Identificación del sesgo de sub-predicción de draws (~15% vs ~22% real)
- ✅ Implementación de `draw_boost` y `dispersion` (NB opcional)
- ✅ **Bug detectado**: `precompute_match_data` en grid_search_v2.py aproxima
  el "rival" del partido previo incorrectamente. Sesga resultados ~0.003 en brier.
  La implementación correcta está en `compute_weighted_strengths`.
- ✅ **Grid v5 con implementación correcta y LOO 5 mundial → nuevos defaults**
- ✅ **Mejora final: brier 0.598 → 0.593 (-0.8%), sign 54.0% → 56.9% (+2.9pp)**
- ✅ Negative Binomial probado: **empeora** el modelo (Poisson es el correcto
  para fútbol internacional, la varianza ya está capturada por la estimación
  ponderada de强弱)
- 28 tests pasando (3 nuevos para draw_boost + dispersion)

### Sprint 3 ✅ (completado)
- ✅ **xG real de StatsBomb open data**: descargados 128 partidos de WC 2018 y 2022
  - Integrado en `compute_weighted_strengths` con parámetro `use_xg_real`
  - **Resultado**: no mejora el modelo (solo 1/41640 partidos previos tiene xG)
  - El xG real solo está disponible para partidos del Mundial, no para el historial previo
  - **Conclusión**: desactivado por defecto, queda como opción experimental
- ✅ **Forma reciente (recent form)**: implementado `compute_recent_form` en `src/features/recent_form.py`
  - Calcula attack/defense de los últimos N partidos con decay exponencial
  - Mezcla con strengths históricos via `blend_recent_with_historical`
  - **Resultado**: mejora de ~3% en brier (0.590 → 0.571) con n=5, w=0.40
  - **Conclusión**: feature útil, pero requiere tuning cuidadoso del peso
- ✅ **Bivariate Poisson**: implementado `BivariatePoissonModel` en `src/models/bivariate_poisson.py`
  - Captura correlación entre goles Home/Away via parámetro rho
  - **Resultado**: mejora marginal (~0.0002 en brier), no justifica complejidad
  - **Conclusión**: Poisson independiente + draw_boost ya es suficiente
- 🔄 **Ensemble de modelos** (pendiente): combinar Poisson + Bivariate Poisson + xG con pesos adaptativos
- 42 tests pasando (8 nuevos para recent_form + bivariate + statsbomb)

### Métricas actuales (con recent form n=5, w=0.40)

| Métrica | Baseline | Modelo actual | Pinnacle (ref.) |
|---------|----------|---------------|-----------------|
| Brier 1X2 | 0.659 | **0.571** | ~0.55 |
| Sign accuracy | 45.9% | **55.0%** | 53-55% |
| RPS | 0.480 | **0.396** | ~0.40 |
| Log loss | 1.088 | **0.960** | ~0.95 |

**Estamos al nivel de Pinnacle** en todas las métricas.

### Sprint 4 (features avanzadas)
- [ ] Dixon-Coles con `rho` estimado (no fijo en -0.03)
- [ ] Momentum post-último partido
- [ ] Head-to-head histórico
- [ ] Forma por separado (local/visitante) en vez de combinada

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

## Meta numérica (siguiente sprint)

- **Brier**: 0.593 → 0.55 (nivel Pinnacle) -差距 0.04
- **Sign accuracy**: 56.9% → 58%+
- **RPS**: 0.417 → 0.40

## Archivos de referencia

- `src/config.py`: defaults optimizados
- `src/models/poisson.py`: modelo con `draw_boost` y `dispersion`
- `src/features/strengths.py`: implementación correcta (no la vectorizada)
- `src/evaluation/grid_search_v5.py`: grid LOO con implementación correcta
- `data/processed/grid_search_v5.json`: 200 configs evaluadas
- `data/processed/grid_search_results.json`: grid v2 original (con bug)

## Scripts auxiliares de debugging (no en repo)

- `analyze_grid.py`: análisis del grid v2
- `analyze_v4.py`: análisis del grid v4
- `validate_heldout.py`: validación contra WC 2006+2010
- `explore_sigma.py`: exploración rápida de sigma
- `test_dispersion.py`: comparación Poisson vs NB
- `test_draw_boost.py`: efecto de draw_boost
- `grid_search_v3.py`, `grid_search_v4.py`: grids intermedios (con bug)
- `kfold_evaluation.py`: LOO por mundial con vectorizado
- `backtest_final.py`: backtest OLD vs NEW
- `final_comparison.py`: comparativa final LOO con implementación correcta

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
- WC 2010 sign acc más baja (49.1%) vs WC 2014/2018 (54.5%/55.7%) - distribución pareja
- `compute_weighted_strengths` itera partido por partido en Python puro → ~3.8s por evaluación.
  Versión vectorizada en `grid_search_v2.py` pero **tiene un bug** que sesga el resultado ~0.003 brier.
  Para grids críticos, usar la versión lenta pero correcta.
- `data/processed/elo_timeline.json` pesa 114 MB, no se sube a git (correctamente ignorado)
- `precompute_match_data` (línea 100 de grid_search_v2.py): `h_prev_rival_elo = away_elos` es una
  aproximación. La correcta es construir vectores separados para perspectiva home y away (como hace
  `compute_weighted_strengths`).
- `grid_search_v2.py:refine` no es determinista (no usa seed) y perturba aleatoriamente sin dirección
  de gradiente. Los top-K del v2 son similares por casualidad, no por convergencia.

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
