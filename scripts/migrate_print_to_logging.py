"""Script one-off para migrar print() a logger.info() en archivos de src/.

NO es parte del paquete, solo para uso de mantenimiento.

Uso:
    python scripts/migrate_print_to_logging.py
"""
from __future__ import annotations

import re
from pathlib import Path

LOGGER_IMPORT = (
    "from src.logging_config import get_logger\n"
    "\n"
    "logger = get_logger(__name__)\n"
)


def migrate_file(path: Path) -> tuple[int, bool]:
    """Migra print() a logger.info() en un archivo.

    Returns:
        (n_replacements, added_import).
    """
    text = path.read_text(encoding="utf-8")
    if "logger = get_logger(__name__)" in text:
        return 0, False  # Ya migrado

    # Reemplazar print(...) por logger.info(...).
    # Usamos regex para encontrar print(...) con cualquier argumento.
    # Patron: print(...) donde (...) puede contener parentesis anidados.
    n = 0
    out = []
    i = 0
    while i < len(text):
        m = re.search(r"(?<![A-Za-z0-9_])print\s*\(", text[i:])
        if not m:
            out.append(text[i:])
            break
        # Encontrar el print, agregar todo antes al output
        out.append(text[i:i + m.start()])
        # Encontrar el parentesis de cierre correspondiente
        paren_start = i + m.end() - 1  # posicion del (
        depth = 1
        j = paren_start + 1
        in_str = None
        while j < len(text) and depth > 0:
            c = text[j]
            if in_str:
                if c == "\\":
                    j += 2
                    continue
                if c == in_str:
                    in_str = None
            else:
                if c in ("'", '"'):
                    in_str = c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth != 0:
            # No se encontro cierre, abortar
            return n, False
        # text[i + m.start(): j + 1] es el print(...) completo
        full_print = text[i + m.start():j + 1]
        # Quitar flush=True de kwargs
        # print(..., flush=True) -> print(...) sin flush
        inner = full_print[6:-1].strip()  # sin "print(" y ")"
        if "flush=True" in inner:
            inner = re.sub(r",\s*flush\s*=\s*True", "", inner)
            if inner.endswith(","):
                inner = inner[:-1].rstrip()
        # Reemplazar por logger.info(...)
        replacement = f"logger.info({inner})"
        out.append(replacement)
        n += 1
        i = j + 1

    new_text = "".join(out)

    # Agregar import + logger = get_logger(...) despues de los docstrings/imports
    if "from src.logging_config import get_logger" not in new_text:
        # Encontrar el ultimo import
        lines = new_text.split("\n")
        last_import = -1
        for k, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last_import = k
        if last_import >= 0:
            lines.insert(last_import + 1, "")
            lines.insert(last_import + 2, "from src.logging_config import get_logger")
            lines.insert(last_import + 3, "")
            lines.insert(last_import + 4, "logger = get_logger(__name__)")
            new_text = "\n".join(lines)
            added = True
        else:
            added = False
    else:
        added = False

    path.write_text(new_text, encoding="utf-8")
    return n, added


def main() -> None:
    src_dir = Path("src")
    files = sorted(src_dir.rglob("*.py"))
    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        n_print = len(re.findall(r"(?<![A-Za-z0-9_])print\s*\(", text))
        if n_print == 0:
            continue
        if "logger = get_logger(__name__)" in text:
            print(f"  {f}: ya migrado, skip")
            continue
        n, added = migrate_file(f)
        rel = str(f).replace("\\", "/")
        print(f"  {rel}: {n} prints -> logger.info (import added: {added})")
        total += n
    print(f"\nTotal: {total} print() -> logger.info() conversions")


if __name__ == "__main__":
    main()
