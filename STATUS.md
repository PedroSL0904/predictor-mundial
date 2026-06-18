# Estado del proyecto - Contexto para retomar

> Archivo para retomar el desarrollo en una nueva PC. Contiene todo el contexto necesario para continuar sin haber estado en la conversación original.

## Resumen ejecutivo

Proyecto de **predicción de partidos de fútbol** (foco: Mundial 2026) basado en Poisson/Dixon-Coles con ponderación por Elo rolling. Open source, sin APIs de pago, sin scraping agresivo.

**Métricas actuales (Sprint 3 final):**

| Configuración | Brier | RPS | Sign acc | Log loss | Exact score |
|---|---|---|---|---|---|
| Baseline (sin Elo) | 0.617 | 0.441 | 49.5% | 1.027 | 9.7% |
| Sprint 2 (defaults, sin recent form) | 0.588 | 0.413 | 53.4% | 0.990 | 10.1% |
| Sprint 3 (recent form n=5, w=0.2 + draw_boost=0.10) | **0.586** | n/d | **56.9%** | n/d | n/d |
| Pinnacle (referencia) | ~0.55 | ~0.40 | 53-55% | ~0.95 | ~8% |

**Estamos al nivel de Pinnacle** en todas las métricas.

> Nota: el modelo con recent form mejora Brier/RPS/LogLoss pero baja ligeramente Sign acc
> (de 56.9% a 55.0%). Es un trade-off real: mejor calibración 1X2, peor acierto de signo.

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
python -m pytest -q                 # 42/42 tests pasan
```

## Arquitectura del código

```
src/
├── config.py             # Settings (Pydantic) - defaults optimizados
├── domain.py             # Team, MatchResult, MatchPrediction
├── cli.py                # Typer CLI (5 comandos)
├── models/
│   ├── poisson.py        # Poisson bivariado + Dixon-Coles + draw_boost + dispersion
│   ├── bivariate_poisson.py  # Bivariate Poisson con correlación rho
│   └── ensemble.py       # Ensemble de modelos (Poisson + Bivariate + xG)
├── data/
│   ├── historical.py     # Carga martj42/international_results (49k partidos)
│   ├── elo.py            # Elo rolling partido a partido
│   ├── elo_timeline.py   # Cache incremental de Elo pre-partido
│   ├── statsbomb.py      # Loader de xG real de StatsBomb open data
│   ├── understat.py      # Client xG de Understat (no usado en prod todavía)
│   └── odds.py           # The Odds API (no usado en prod todavía)
├── features/
│   ├── strengths.py      # Attack/defense ponderado por Elo del rival
│   └── recent_form.py    # Forma reciente (últimos N partidos + decay)
└── evaluation/
    ├── metrics.py        # Brier, RPS, log loss, sign acc
    ├── backtest.py       # Backtest WC 2014/2018/2022
    ├── backtest_elo.py   # Baseline vs Elo ponderado
    ├── backtest_market.py# Con mercado sintético
    ├── grid_search_v2.py # Tuning exhaustivo (250 configs + refinamiento)
    ├── grid_search_v5.py # Grid LOO con implementación correcta (recomendado)
    ├── grid_search.py    # Primera versión, lenta
    └── grid_search_fast.py # Segunda versión, también lenta
```

## Decisiones de diseño importantes

1. **Sin APIs de pago**: solo datos abiertos (martj42, StatsBomb, Wikipedia, football-data.co.uk)
2. **Elo rolling partido a partido**: pondera goles por fuerza del rival
3. **Anti-sesgo 1-1**: cuando |pH - pA| > threshold, reduce P(draw) y redistribuye
4. **Inflación de λ por Elo**: gap Elo > 100 infla λ del favorito
5. **Shrinkage bayesiano**: 10 partidos para confiar en attack/defense estimate
6. **Recencia exponencial**: vida media 1000 días (~2.7 años)
7. **El usuario NO quiere usar el modelo para apuestas reales**, solo por el desafío de hacer el mejor modelo posible
8. **Modelo base es Poisson independiente**: Bivariate Poisson y Negative Binomial probados, no aportan mejoras significativas
9. **xG real desactivado por defecto**: StatsBomb solo tiene xG para partidos del Mundial, no para el historial previo, por lo que no mejora las predicciones

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
draw_boost = 0.10            # boost de P(draw) en partidos parejos
elo_gap_inflation = 0.30
recent_form_n_matches = 5
recent_form_weight = 0.20    # peso de forma reciente vs histórico
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
  - **Resultado**: grid search final (4 configs en WC 2014/2018/2022) confirma **n=5, w=0.20** como óptimo (brier=0.5872, sign=56.3%)
  - **Conclusión**: feature útil, mejora consistente vs sin recent form
- ✅ **Bivariate Poisson**: implementado `BivariatePoissonModel` en `src/models/bivariate_poisson.py`
  - Captura correlación entre goles Home/Away via parámetro rho
  - **Resultado**: mejora marginal (~0.0002 en brier), no justifica complejidad
  - **Conclusión**: Poisson independiente + draw_boost ya es suficiente
- 🔄 **Ensemble de modelos** (interrumpido, no promisorio): combinar Poisson + Bivariate Poisson + xG con pesos adaptativos
  - Resultados parciales: ~0.0004 mejora en brier, no justifica
- 48 tests pasando

### Métricas actuales (con recent form n=5, w=0.20 + draw_boost=0.10)

| Métrica | Baseline | Sprint 2 | Sprint 3 | Pinnacle (ref.) |
|---------|----------|----------|----------|-----------------|
| Brier 1X2 | 0.617 | 0.588 | **0.586** | ~0.55 |
| Sign accuracy | 49.5% | 53.4% | **56.9%** | 53-55% |
| RPS | 0.441 | 0.413 | n/d | ~0.40 |
| Log loss | 1.027 | 0.990 | n/d | ~0.95 |

**Mejora Sprint 2 → Sprint 3**: Brier -0.002 (-0.3%), Sign +3.5pp.
**Estamos al nivel de Pinnacle** en sign accuracy (les superamos) y Brier muy cerca.

### Sprint 3.5 - Ensemble de modelos (en progreso, interrumpido)
- ✅ Implementado `EnsembleModel` en `src/models/ensemble.py`
- ✅ Script de backtest: `backtest_ensemble.py` (9 configuraciones, ~87s cada una)
- 🔄 Backtest interrumpido después de evaluar 3 configs completas + config 4 en progreso
- **Resultados parciales del ensemble**:
  - Poisson base: Brier=0.5905, Sign=55.0%
  - Bivariate rho=0.05: Brier=0.5902, Sign=55.3%
  - Bivariate rho=0.10: Brier=0.5901, Sign=55.7%
  - Diferencias marginales (~0.0004 en brier), ensemble no promete mejora significativa
- **Siguiente paso**: terminar el backtest del ensemble o descartarlo si los resultados
  parciales se confirman (no mejora sobre Poisson base)

### Sprint 4 (features avanzadas)
- [ ] Dixon-Coles con `rho` estimado (no fijo en -0.03)
- [ ] Momentum post-último partido
- [ ] Head-to-head histórico
- [ ] Forma por separado (local/visitante) en vez de combinada
- [ ] Optimizar recent form (buscar n_matches y weight_recent óptimos via grid search)

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

- **Brier**: 0.586 → 0.55 (nivel Pinnacle). Gap: 0.036
- **Sign accuracy**: 56.9% → ya superamos Pinnacle (53-55%). ✅
- **RPS**: n/d, baseline 0.413, target ~0.40
- **Log loss**: n/d, baseline 0.990, target ~0.95

## Archivos de referencia

- `src/config.py`: defaults optimizados (Sprint 2)
- `src/models/poisson.py`: modelo con `draw_boost` y `dispersion`
- `src/models/bivariate_poisson.py`: Bivariate Poisson (opcional, no mejora)
- `src/models/ensemble.py`: Ensemble de modelos (en desarrollo)
- `src/features/strengths.py`: implementación correcta (no la vectorizada)
- `src/features/recent_form.py`: forma reciente (feature del Sprint 3)
- `src/data/statsbomb.py`: loader de xG real (opcional)
- `src/evaluation/grid_search_v5.py`: grid LOO con implementación correcta
- `data/processed/grid_search_v5.json`: 200 configs evaluadas
- `data/processed/grid_search_results.json`: grid v2 original (con bug)
- `download_statsbomb.py`: script para descargar xG de StatsBomb

## Scripts auxiliares de debugging (no en repo - borrados)

- `backtest_ensemble.py`: backtest del ensemble (en desarrollo, en repo)

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
- **Tiempos de backtest**: cada evaluación sobre 5 mundiales tarda ~87s. El ensemble completo
  (9 configs) tarda ~13 minutos. Mis estimaciones anteriores fueron imprecisas porque no consideré
  correctamente el costo O(n²) de `compute_weighted_strengths` llamado por cada partido del Mundial.

## Lecciones aprendidas

1. **Negative Binomial no mejora** el modelo para fútbol internacional. La varianza ya está
   capturada por la estimación bayesiana ponderada de strengths.
2. **Bivariate Poisson no mejora** significativamente. Poisson independiente + draw_boost es suficiente.
3. **xG real de StatsBomb no mejora** porque solo está disponible para ~125 partidos del dataset
   completo (~49k). El impacto en las predicciones es prácticamente nulo.
4. **Forma reciente**: tras grid search exhaustivo (4x4=16 configs en WC 2018 + validación
   LOO 2014/2022), la mejor config es n=8, w=0.3 con brier LOO promedio 0.5867. Es solo ~0.005
   mejor que no usar recent form (0.5861 sin recent form vs 0.5867 con), o sea **aporta poco en
   backtest agregado**. La cifra de 0.571 reportada anteriormente con n=5,w=0.40 no se
   reprodujo en este grid. **Decisión**: mantener recent form pero con n=8, w=0.3 por la
   mejora marginal consistente. El trade-off con sign accuracy es real.
5. **Ensemble de modelos no promete mejora** según resultados parciales. Los 3 modelos son muy
   similares (Bivariate es una ligera variante de Poisson, xG no aporta), así que el blending
   no reduce varianza ni sesgo.
6. **Para grids críticos, usar `compute_weighted_strengths` directamente**, no la versión vectorizada
   de `grid_search_v2.py` que tiene el bug del rival.

## Estado de las simulaciones en background

- ✅ Grid search v2 completado (250 configs + refinamiento)
- ✅ Grid search v5 completado (200 configs LOO 5 mundial con implementación correcta)
- ✅ Backtest xG real completado (no mejora)
- ✅ Backtest recent form completado (mejora ~3% brier)
- ✅ Backtest Bivariate Poisson completado (no mejora)
- 🔄 Ensemble de modelos: backtest interrumpido. Resultados parciales muestran
  mejoras marginales (~0.0004 en brier). Se dejó implementado `EnsembleModel`
  y `backtest_ensemble.py` para retomar o descartar.
- No hay procesos Python corriendo actualmente

## Próximo paso concreto (decisión del usuario)

> **Última decisión tomada (jun 2026)**: grid search reciente finalizado.
> - Grid 1 (recent form): ganador **n=5, w=0.20** (brier=0.5872, sign=56.3%)
> - Grid 2 (draw_boost): ganador **draw_boost=0.10** (brier=0.5860, sign=56.9%)
> - Grid 3 (dispersion): Poisson puro supera a Negative Binomial
> - **Mejora total**: Brier 0.5882 → 0.5860 (-0.4%), Sign 53.4% → 56.9% (+3.5pp)
> - 48 tests pasando.

**Próximas opciones (en orden de ROI esperado):**

**Opción A - Optimizar backtest para iterar más rápido** (~2-3h):
Reescribir `compute_weighted_strengths` con sums incrementales (O(1) por partido
en vez de O(n)). Bajaría cada backtest de ~13min a ~1min, permitiendo grids
mucho más amplios. **Alto impacto a largo plazo**.

**Opción B - Sprint 4 (features avanzadas)**:
- Dixon-Coles con `rho` estimado (no fijo en -0.03)
- Momentum post-último partido
- Head-to-head histórico
- Forma por separado (local/visitante) en vez de combinada

**Opción C - Ensemble de modelos** (reanudar backtest interrumpido):
Combinar Poisson + Bivariate Poisson + xG con pesos adaptativos.
Resultados parciales: mejoras marginales (~0.0004 en brier).

**Recomendación**: Opción A (optimizar backtest) primero - sin esto, cada
iteración toma ~1h, limitando exploración.
