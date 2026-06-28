"""Sprint A4b: re-optimiza ensemble LOO con features historicas ACTIVAS.

Compara los pesos optimos y brier con/sin features para mostrar
si la inclusion de H2H + momentum + WC history cambia el optimo.
"""
from __future__ import annotations

import time
from pathlib import Path

from src.data.elo_timeline import precompute_and_cache
from src.data.historical import load_martj42_csv
from src.evaluation.ensemble_optimization import loo_optimize_ensemble
from src.features.strengths_cache import StrengthsCache
from src.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    csv_path = Path("data/raw/martj42_results.csv")
    cache_path = Path("data/processed/elo_timeline.parquet")
    logger.info("Cargando datos...")
    timeline = precompute_and_cache(csv_path, cache_path)
    df = load_martj42_csv(csv_path)
    cache = StrengthsCache(df, timeline)

    # Baseline: sin features (Sprint A4 original)
    logger.info("\n=== LOO 3 mundial SIN features (Sprint A4) ===")
    t0 = time.time()
    weights_off = loo_optimize_ensemble(df, cache, timeline, enable_historical_features=False)
    logger.info(f"Pesos: {weights_off}")
    logger.info(f"Tiempo: {time.time()-t0:.1f}s")

    # Con features (Sprint A4b)
    logger.info("\n=== LOO 3 mundial CON features (Sprint A4b) ===")
    t0 = time.time()
    weights_on = loo_optimize_ensemble(df, cache, timeline, enable_historical_features=True)
    logger.info(f"Pesos: {weights_on}")
    logger.info(f"Tiempo: {time.time()-t0:.1f}s")

    # Comparativa
    logger.info("\n=== Comparativa ===")
    logger.info(f"  SIN features: P={weights_off.poisson:.2f}, BP={weights_off.bivariate_poisson:.2f}, S={weights_off.skellam:.2f}, brier={weights_off.brier_train:.4f}")
    logger.info(f"  CON features: P={weights_on.poisson:.2f}, BP={weights_on.bivariate_poisson:.2f}, S={weights_on.skellam:.2f}, brier={weights_on.brier_train:.4f}")
    improvement = (weights_off.brier_train - weights_on.brier_train) / weights_off.brier_train * 100
    logger.info(f"  Mejora brier: {improvement:+.2f}%")

    # Guardar reporte
    out_path = Path("data/processed/ensemble_loo_with_features.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = f"""# Ensemble LOO: con vs sin features historicas

## Sin features (Sprint A4 original)
- Pesos optimos: P={weights_off.poisson:.2f}, BP={weights_off.bivariate_poisson:.2f}, S={weights_off.skellam:.2f}
- Brier promedio LOO: {weights_off.brier_train:.4f}

## Con features (Sprint A4b)
- Pesos optimos: P={weights_on.poisson:.2f}, BP={weights_on.bivariate_poisson:.2f}, S={weights_on.skellam:.2f}
- Brier promedio LOO: {weights_on.brier_train:.4f}

## Mejora
- Delta brier: {weights_on.brier_train - weights_off.brier_train:+.4f}
- Mejora porcentual: {improvement:+.2f}%

## Conclusion
"""
    if improvement > 0:
        md += "Las features mejoran el ensemble LOO.\n"
    else:
        md += "Las features NO mejoran el ensemble LOO (o lo empeoran marginalmente).\n"
    md += f"\nRecomendacion: usar pesos = [{weights_on.poisson:.2f}, {weights_on.bivariate_poisson:.2f}, {weights_on.skellam:.2f}]\n"

    out_path.write_text(md, encoding="utf-8")
    logger.info(f"\nReporte guardado en {out_path}")


if __name__ == "__main__":
    main()
