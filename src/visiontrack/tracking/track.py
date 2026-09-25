"""Track state: what the tracker remembers about one object across frames."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..detection import BBox, Detection, Point
from ..vision.geometry import cardinal_direction, distance


@dataclass(slots=True)
class Track:
    """A tracked object.

    Attributes:
        track_id: Stable identifier for this track's lifetime.
        centroid: Current centroid.
        previous_centroid: Centroid at the previous *matched* frame (``None`` on the first match).
        bbox: Current bounding box.
        first_seen: Stream time the track was created.
        last_seen: Stream time of the most recent matched detection.
        last_seen_frame: Frame number of the most recent matched detection.
        disappeared_frames: Consecutive frames since the last matched detection.
        matched_observations: Count of frames this track has been matched to a detection.
        velocity: Smoothed ``(vx, vy)`` in pixels/second. Pixel units, not physical distance —
            only meaningful with camera calibration, which VisionTrack does not perform.
        zones: Zone names this track is currently confirmed inside (updated externally by the
            zone-occupancy tracker; kept here so overlay/events have one place to read it).
        label: Class label, for tracks initiated from ML detections.
        source: Detection source that created this track (``"motion"`` or ``"ml"``).
    """

    track_id: int
    centroid: Point
    bbox: BBox
    first_seen: float
    last_seen: float
    last_seen_frame: int
    previous_centroid: Point | None = None
    disappeared_frames: int = 0
    matched_observations: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    zones: set[str] = field(default_factory=set)
    label: str | None = None
    source: str = "motion"

    @property
    def speed(self) -> float:
        """Scalar speed in pixels/second."""
        return distance((0.0, 0.0), self.velocity)

    @property
    def direction(self) -> str:
        """Coarse movement direction: up/down/left/right/stationary."""
        return cardinal_direction(*self.velocity, min_speed=1.0)

    @property
    def age_s(self) -> float:
        """Time since the track was created, in seconds of stream time."""
        return self.last_seen - self.first_seen

    def apply_match(self, detection: Detection, timestamp: float, frame_number: int, alpha: float) -> None:
        """Update state from a matched detection using an exponential moving average for velocity."""
        dt = timestamp - self.last_seen
        new_vel = (0.0, 0.0)
        if dt > 0:
            new_vel = (
                (detection.centroid[0] - self.centroid[0]) / dt,
                (detection.centroid[1] - self.centroid[1]) / dt,
            )
        self.previous_centroid = self.centroid
        self.velocity = (
            alpha * new_vel[0] + (1 - alpha) * self.velocity[0],
            alpha * new_vel[1] + (1 - alpha) * self.velocity[1],
        )
        self.centroid = detection.centroid
        self.bbox = detection.bbox
        self.last_seen = timestamp
        self.last_seen_frame = frame_number
        self.disappeared_frames = 0
        self.matched_observations += 1
        if detection.label is not None:
            self.label = detection.label

    def mark_disappeared(self) -> None:
        """Record one frame with no matching detection."""
        self.disappeared_frames += 1
