"""Debounced zone occupancy tracking per object track.

Raw per-frame membership is noisy at a region boundary. :class:`ZoneOccupancyTracker` requires
``enter_frames``/``exit_frames`` consecutive observations before it reports a transition, and
tracks dwell time so :mod:`visiontrack.events.rules` can derive dwell events deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class _TrackZoneState:
    inside: bool = False
    consecutive_in: int = 0
    consecutive_out: int = 0
    entered_at: float | None = None  # stream time the zone was confirmed entered


@dataclass(slots=True)
class ZoneTransition:
    """One confirmed enter/leave event for a track in a zone."""

    track_id: int
    zone: str
    entered: bool  # True = entered, False = left
    timestamp: float
    dwell_seconds: float | None = None  # set only when entered=False


class ZoneOccupancyTracker:
    """Debounced enter/leave state machine, one instance per pipeline (all zones, all tracks)."""

    def __init__(self, enter_frames: int = 2, exit_frames: int = 2) -> None:
        self._enter_frames = enter_frames
        self._exit_frames = exit_frames
        self._state: dict[tuple[int, str], _TrackZoneState] = {}

    def update(
        self, track_id: int, zone: str, raw_inside: bool, timestamp: float
    ) -> ZoneTransition | None:
        """Feed one frame's raw membership for ``(track_id, zone)``; returns a confirmed transition, if any."""
        key = (track_id, zone)
        state = self._state.setdefault(key, _TrackZoneState())

        if raw_inside:
            state.consecutive_in += 1
            state.consecutive_out = 0
        else:
            state.consecutive_out += 1
            state.consecutive_in = 0

        if not state.inside and raw_inside and state.consecutive_in >= self._enter_frames:
            state.inside = True
            state.entered_at = timestamp
            return ZoneTransition(track_id, zone, entered=True, timestamp=timestamp)

        if state.inside and not raw_inside and state.consecutive_out >= self._exit_frames:
            state.inside = False
            dwell = timestamp - state.entered_at if state.entered_at is not None else None
            state.entered_at = None
            return ZoneTransition(track_id, zone, entered=False, timestamp=timestamp, dwell_seconds=dwell)

        return None

    def is_inside(self, track_id: int, zone: str) -> bool:
        """Confirmed (debounced) occupancy, ignoring in-flight debounce counters."""
        state = self._state.get((track_id, zone))
        return state.inside if state else False

    def dwell_seconds(self, track_id: int, zone: str, timestamp: float) -> float | None:
        """Elapsed dwell time so far for a track currently confirmed inside a zone."""
        state = self._state.get((track_id, zone))
        if state is None or not state.inside or state.entered_at is None:
            return None
        return timestamp - state.entered_at

    def zones_for_track(self, track_id: int) -> list[str]:
        """Zones the track is currently confirmed inside."""
        return [zone for (tid, zone), s in self._state.items() if tid == track_id and s.inside]

    def forget_track(self, track_id: int) -> list[ZoneTransition]:
        """Remove a track's state (called on track removal); returns synthetic LEFT transitions
        for any zone it was still confirmed inside, so dwell/zone bookkeeping stays consistent.
        """
        transitions = []
        for key in [k for k in self._state if k[0] == track_id]:
            state = self._state.pop(key)
            if state.inside:
                transitions.append(
                    ZoneTransition(track_id, key[1], entered=False, timestamp=0.0, dwell_seconds=None)
                )
        return transitions
