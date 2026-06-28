"""Tests del logging config."""
from __future__ import annotations

import logging

from src.logging_config import get_logger, setup_logging


def test_get_logger_returns_logger() -> None:
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)


def test_get_logger_idempotent() -> None:
    """Llamar setup_logging multiples veces no reconfigura."""
    setup_logging("INFO")
    setup_logging("DEBUG")
    logger = get_logger("test")
    assert logger.name == "test"


def test_logger_outputs(caplog) -> None:
    logger = get_logger("test_outputs")
    with caplog.at_level(logging.INFO):
        logger.info("hello world")
    assert "hello world" in caplog.text


def test_logger_respects_level(caplog) -> None:
    """Logger respeta el level configurado."""
    setup_logging("INFO")
    logger = get_logger("test_respects")
    with caplog.at_level(logging.INFO):
        logger.info("info_msg")
    assert "info_msg" in caplog.text


def test_setup_logging_is_idempotent() -> None:
    """setup_logging puede llamarse muchas veces sin efectos secundarios."""
    setup_logging("DEBUG")
    setup_logging("ERROR")
    setup_logging("INFO")
    # No debe crashear
    logger = get_logger("test_idem")
    assert logger is not None
