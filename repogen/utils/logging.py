"""Centralised logging setup for all RepoGen modules.

Every module in the package should use this instead of configuring
its own logger.  Calling ``setup_logging`` twice with the same *name*
returns the existing logger without adding duplicate handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_CONFIGURED_LOGGERS: set[str] = set()


def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a named logger.

    If *log_file* is provided the logger also writes to that file.
    Subsequent calls with the same *name* return the cached logger so
    handlers are never duplicated.

    Args:
        name: Logger name - typically ``__name__`` of the calling module.
        level: Logging level as a string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    numeric_level = getattr(logging, level.upper(), None)
    if numeric_level is None:
        raise ValueError(
            f"Invalid log level: '{level}'. "
            f"Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    logger.setLevel(numeric_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(_FORMATTER)
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger


def reconfigure_logging(level: str) -> None:
    """Update the log level of all ``repogen.*`` loggers.

    Call this after loading :class:`PipelineConfig` to propagate the
    configured log level to every logger that was created at import time.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Raises:
        ValueError: If *level* is not a valid logging level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if numeric_level is None:
        raise ValueError(
            f"Invalid log level: '{level}'. "
            f"Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    for name, logger_obj in logging.Logger.manager.loggerDict.items():
        if name.startswith("repogen") and isinstance(logger_obj, logging.Logger):
            logger_obj.setLevel(numeric_level)
            for handler in logger_obj.handlers:
                handler.setLevel(numeric_level)
