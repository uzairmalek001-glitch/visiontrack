"""Event data models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..detection import BBox

_FLOAT_DECIMALS = 4


class EventType(StrEnum):
    """All event types the engine can emit."""

    MOTION_STARTED = "MOTION_STARTED"
    MOTION_ACTIVE = "MOTION_ACTIVE"
    MOTION_STOPPED = "MOTION_STOPPED"
    OBJECT_ENTERED_ZONE = "OBJECT_ENTERED_ZONE"
    OBJECT_LEFT_ZONE = "OBJECT_LEFT_ZONE"
    OBJECT_DWELL_STARTED = "OBJECT_DWELL_STARTED"
    OBJECT_DWELL_ENDED = "OBJECT_DWELL_ENDED"
    LINE_CROSSED = "LINE_CROSSED"
    VELOCITY_EXCEEDED = "VELOCITY_EXCEEDED"
    ZONE_ACTIVATED = "ZONE_ACTIVATED"
    ZONE_DEACTIVATED = "ZONE_DEACTIVATED"


@dataclass(slots=True)
class EventDraft:
    """What a rule produces; the engine stamps id/time/frame/source to make it an :class:`Event`."""

    event_type: EventType
    track_id: int | None = None
    zone: str | None = None
    bbox: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Event:
    """A structured event produced by a state transition.

    Attributes:
        event_id: Sequential, deterministic identifier (``evt-000001``...).
        event_type: What happened.
        timestamp: Stream time in seconds (video time for files, elapsed monotonic time for live
            sources). Deterministic for a given input file.
        frame_number: Index of the source frame that triggered the event.
        source: Source label, e.g. ``camera:0`` or the video file's base name.
        track_id: Track involved, if any.
        zone: Zone name, if any.
        bbox: ``(x, y, w, h)`` of the involved object/region, if any.
        metadata: Event-specific details (durations, speeds, line name, ...).
        wall_time: Optional ISO-8601 UTC wall-clock time; only set when explicitly enabled
            because it makes output non-reproducible.
    """

    event_id: str
    event_type: EventType
    timestamp: float
    frame_number: int
    source: str
    track_id: int | None = None
    zone: str | None = None
    bbox: BBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    wall_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict with a stable key order; ``None`` optionals are omitted."""
        data: dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": str(self.event_type),
            "timestamp": _round(self.timestamp),
            "frame_number": self.frame_number,
            "source": self.source,
        }
        if self.wall_time is not None:
            data["wall_time"] = self.wall_time
        if self.track_id is not None:
            data["track_id"] = self.track_id
        if self.zone is not None:
            data["zone"] = self.zone
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        if self.metadata:
            data["metadata"] = _round(self.metadata)
        return data

    def to_json(self) -> str:
        """Compact single-line JSON (one JSONL record)."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Inverse of :meth:`to_dict` (used by replay comparison)."""
        bbox = data.get("bbox")
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=float(data["timestamp"]),
            frame_number=int(data["frame_number"]),
            source=data["source"],
            track_id=data.get("track_id"),
            zone=data.get("zone"),
            bbox=tuple(bbox) if bbox is not None else None,  # type: ignore[arg-type]
            metadata=dict(data.get("metadata", {})),
            wall_time=data.get("wall_time"),
        )


def _round(value: Any) -> Any:
    """Recursively round floats so serialised output is stable across platforms."""
    if isinstance(value, float):
        return round(value, _FLOAT_DECIMALS)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value
