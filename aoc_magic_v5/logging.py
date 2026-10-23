"""Logging helpers for aoc_magic_v3."""

from __future__ import annotations

import logging
from typing import Optional


_FORMAT = "[%(levelname)s] %(name)s: %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name* using the package format."""
    return logging.getLogger(name)


def configure_logging(level: str = "info") -> None:
    """Configure root handlers for cli/gui. Idempotent."""
    global _configured
    if _configured:
        root = logging.getLogger()
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    _configured = True
