"""Setup de datos para CI y primera ejecucion local.

Descarga:
- martj42/international_results (CSV principal, ~3.7 MB, ~49k partidos)
- statsbomb_xg.json (lookup agregado, 25 KB) - solo si no existe

NO descarga eventos crudos de StatsBomb (~210 MB). Esos se generan
localmente con download_statsbomb.py y NO se commitean.

Uso:
    python scripts/setup_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

MARTJ42_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/"
    "results.csv"
)

CSV_DEST = Path("data/raw/martj42_results.csv")


def _download(url: str, dest: Path, chunk_mb: int = 1) -> None:
    """Descarga un archivo con barra de progreso basica."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  Ya existe: {dest} ({size_mb:.1f} MB)")
        return

    print(f"  Descargando {url}...", flush=True)
    t0 = time.time()
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100 * downloaded / total
                    print(
                        f"    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                        f"({pct:.0f}%)",
                        end="\r",
                        flush=True,
                    )
    elapsed = time.time() - t0
    print(f"\n  Descargado en {elapsed:.1f}s -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    """Descarga todos los datos necesarios. Retorna 0 si OK, 1 si falla."""
    print("=" * 70)
    print("Setup de datos para predictor-mundial")
    print("=" * 70)

    print("\n[1/2] martj42/international_results (CSV principal)")
    try:
        _download(MARTJ42_URL, CSV_DEST)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    # statsbomb_xg.json se commitea al repo, asi que no necesitamos descargarlo
    sb_path = Path("data/raw/statsbomb_xg.json")
    if sb_path.exists():
        print(f"\n[2/2] StatsBomb xG lookup: ya existe ({sb_path.stat().st_size / 1e3:.1f} KB)")
    else:
        print("\n[2/2] StatsBomb xG lookup: NO encontrado")
        print("       Para regenerarlo, correr: python download_statsbomb.py")
        print("       (toma ~5-10 min, baja ~210 MB de eventos de StatsBomb)")
        # No fallamos: tests que no usan StatsBomb siguen pasando

    print("\n" + "=" * 70)
    print("Setup completo. Datos en data/raw/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
