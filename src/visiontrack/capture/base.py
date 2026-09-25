"""Frame source abstraction shared by camera, video-file, and image-sequence sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured frame.

    Attributes:
        image: BGR (or grayscale) ``uint8`` array as returned by the source.
        frame_number: 0-based index within this source.
        timestamp: Stream time in seconds — ``frame_number / fps`` for files/sequences,
            elapsed monotonic time since the first frame for live sources.
        capture_latency: Wall-clock seconds spent in the underlying read call, if measurable.
    """

    image: np.ndarray
    frame_number: int
    timestamp: float
    capture_latency: float | None = None


class FrameSource(ABC):
    """Common interface for anything that yields frames: camera, video file, image sequence,
    and — without redesigning the architecture — a future network stream.
    """

    @abstractmethod
    def open(self) -> None:
        """Open the underlying resource. Raises :class:`~visiontrack.errors.CaptureError`."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is exhausted (end of file)."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource. Always safe to call, including more than once."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second of the source (best effort for live sources)."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int] | None:
        """``(width, height)`` if known before the first frame is read, else ``None``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label used as the ``source`` field of emitted events (e.g. ``camera:0``)."""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
