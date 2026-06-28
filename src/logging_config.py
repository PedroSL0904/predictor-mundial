"""Configuracion centralizada de logging.

Uso:
    from src.logging_config import get_logger, setup_logging

    logger = get_logger(__name__)
    setup_logging(level="INFO")
    logger.info("Entrenando modelo...")

Output por default: INFO+ a stdout, formato %(message)s (sin timestamp
para mantener output similar a print()).
"""
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(message)s"
_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configura el root logger una sola vez."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retorna un logger. Llama a setup_logging la primera vez."""
    setup_logging()
    return logging.getLogger(name)
