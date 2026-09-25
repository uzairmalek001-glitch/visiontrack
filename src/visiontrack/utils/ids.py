"""Deterministic sequential ID generation (no global state)."""

from __future__ import annotations


class IdGenerator:
    """Hands out consecutive integers starting at ``start``.

    Instances are independent, so two pipelines never share counters and replays are reproducible.
    """

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def next(self) -> int:
        """Return the next ID and advance the counter."""
        value = self._next
        self._next += 1
        return value

    @property
    def peek(self) -> int:
        """The ID that :meth:`next` would return, without consuming it."""
        return self._next


def format_event_id(number: int) -> str:
    """Format a sequence number as a stable event ID, e.g. ``evt-000042``."""
    return f"evt-{number:06d}"
