"""
Structured logging for Skills Agent.

Provides:
  - Rich console output (colored, with module/timestamp)
  - Optional rotating file handler
  - JSON format mode for production
  - Third-party logger quieting
  - One-call setup: `setup_logging(settings.log)`

Call `setup_logging()` once at application startup. All modules use
standard `logging.getLogger(__name__)` — this module configures the
root logger and the `skills_agent` hierarchy.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LogSettings

# Package-level logger namespace
LOGGER_NAME = "skills_agent"

_is_setup = False


def setup_logging(log_settings: LogSettings | None = None) -> logging.Logger:
    """Configure logging for the entire skills_agent package.

    Args:
        log_settings: LogSettings instance. If None, uses defaults.

    Returns:
        The configured root `skills_agent` logger.
    """
    global _is_setup
    if _is_setup:
        return logging.getLogger(LOGGER_NAME)

    if log_settings is None:
        from .config import LogSettings
        log_settings = LogSettings()

    level = getattr(logging, log_settings.level.upper(), logging.INFO)

    # Root package logger
    pkg_logger = logging.getLogger(LOGGER_NAME)
    pkg_logger.setLevel(level)
    pkg_logger.handlers.clear()

    # ---- Console handler ----
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)

    if log_settings.format == "rich":
        console.setFormatter(_RichFormatter(
            show_timestamp=log_settings.show_timestamp,
            show_module=log_settings.show_module,
        ))
    elif log_settings.format == "json":
        console.setFormatter(_JsonFormatter())
    else:
        fmt_parts = []
        if log_settings.show_timestamp:
            fmt_parts.append("%(asctime)s")
        fmt_parts.append("%(levelname)-8s")
        if log_settings.show_module:
            fmt_parts.append("[%(name)s]")
        fmt_parts.append("%(message)s")
        console.setFormatter(logging.Formatter(" ".join(fmt_parts)))

    pkg_logger.addHandler(console)

    # ---- File handler (optional) ----
    if log_settings.file:
        log_path = Path(log_settings.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=log_settings.file_max_bytes,
            backupCount=log_settings.file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        pkg_logger.addHandler(file_handler)

    # ---- Quiet noisy third-party loggers ----
    for name in log_settings.quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    _is_setup = True
    pkg_logger.debug("Logging configured: level=%s format=%s", log_settings.level, log_settings.format)
    return pkg_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the skills_agent namespace.

    Usage in any module:
        from skills_agent.log import get_logger
        logger = get_logger(__name__)
    """
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


# ======================================================================
# Formatters
# ======================================================================

# ANSI color codes (no dependency on `rich`)
_COLORS = {
    "DEBUG":    "\033[36m",     # cyan
    "INFO":     "\033[32m",     # green
    "WARNING":  "\033[33m",     # yellow
    "ERROR":    "\033[31m",     # red
    "CRITICAL": "\033[1;31m",   # bold red
}
_RESET = "\033[0m"
_DIM = "\033[2m"


class _RichFormatter(logging.Formatter):
    """Colored console formatter with optional timestamp and module."""

    def __init__(self, show_timestamp: bool = True, show_module: bool = True):
        super().__init__()
        self.show_timestamp = show_timestamp
        self.show_module = show_module

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        parts = []

        if self.show_timestamp:
            ts = self.formatTime(record, "%H:%M:%S")
            parts.append(f"{_DIM}{ts}{_RESET}")

        parts.append(f"{color}{record.levelname:<8s}{_RESET}")

        if self.show_module:
            # Shorten module path: skills_agent.loader → loader
            short = record.name
            if short.startswith(f"{LOGGER_NAME}."):
                short = short[len(LOGGER_NAME) + 1:]
            parts.append(f"{_DIM}[{short}]{_RESET}")

        parts.append(record.getMessage())

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            parts.append(f"\n{record.exc_text}")

        return " ".join(parts)


class _JsonFormatter(logging.Formatter):
    """JSON-lines formatter for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)
