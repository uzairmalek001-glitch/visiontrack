"""VisionTrack – deterministic-first real-time computer-vision event engine."""

from .detection import BBox, Detection, Point
from .errors import (
    CaptureError,
    ConfigError,
    DetectorUnavailableError,
    FrameError,
    VisionTrackError,
)

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "CaptureError",
    "ConfigError",
    "Detection",
    "DetectorUnavailableError",
    "FrameError",
    "Point",
    "VisionTrackError",
    "__version__",
]
