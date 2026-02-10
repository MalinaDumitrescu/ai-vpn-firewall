from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def _resolve_level(level: str | int | None, default: int) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), default)
    return default


def setup_logger(
    name: str = "ai-vpn-firewall",
    level: str | int = "INFO",
    log_file: Optional[Path] = None,
    console_level: str | int | None = None,
    file_level: str | int | None = None,
) -> logging.Logger:

    logger = logging.getLogger(name)
    base_level = _resolve_level(level, logging.INFO)

    logger.setLevel(base_level)
    logger.propagate = False

    # IMPORTANT: avoid duplicate handlers in notebooks
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(_resolve_level(console_level, base_level))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(_resolve_level(file_level, base_level))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
