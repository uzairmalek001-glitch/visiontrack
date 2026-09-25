"""Human-readable logging setup and an event sink that logs through it."""

from __future__ import annotations

import logging

from ..config import LoggingConfig
from ..events.models import Event

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(config: LoggingConfig) -> logging.Logger:
    """Configure and return the ``visiontrack`` logger. Idempotent (safe to call more than once)."""
    logger = logging.getLogger("visiontrack")
    logger.setLevel(getattr(logging, config.level))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
    else:
        logger.handlers[0].setLevel(getattr(logging, config.level))
    return logger


def console_event_sink(logger: logging.Logger):
    """Build an :data:`~visiontrack.events.engine.EventSink` that logs one line per event."""

    def sink(event: Event) -> None:
        parts = [f"[{event.event_id}]", str(event.event_type), f"t={event.timestamp:.3f}s"]
        if event.track_id is not None:
            parts.append(f"track={event.track_id}")
        if event.zone is not None:
            parts.append(f"zone={event.zone}")
        if event.metadata:
            parts.append(str(event.metadata))
        logger.info(" ".join(parts))

    return sink
