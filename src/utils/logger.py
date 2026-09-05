"""Centralised logging configuration.

Every module obtains its logger through :func:`get_logger` so that formatting,
level and handlers are configured in exactly one place.  The level is driven by
the ``LOG_LEVEL`` environment variable (see :mod:`src.utils.config`).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Guards against attaching duplicate handlers when modules are re-imported
# (Streamlit re-runs the script on every interaction).
_CONFIGURED: set[str] = set()


def _resolve_level() -> int:
    """Read the desired log level from the environment, defaulting to INFO."""
    raw = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    return getattr(logging, raw, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` with a single stdout handler attached.
    """
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(name)
    logger.setLevel(_resolve_level())
    return logger


def set_global_level(level: str | int) -> None:
    """Override the level on every logger this module has configured.

    Args:
        level: Either a level name (``"DEBUG"``) or a numeric logging level.
    """
    resolved = getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else level
    for name in _CONFIGURED:
        logging.getLogger(name).setLevel(resolved)
