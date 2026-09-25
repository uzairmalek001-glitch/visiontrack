"""Exception hierarchy for VisionTrack.

All expected, user-actionable failures derive from :class:`VisionTrackError` so the CLI can
report them with a clean message instead of a traceback.
"""
from __future__ import annotations


class VisionTrackError(Exception):
    """Base class for all expected VisionTrack errors."""


class ConfigError(VisionTrackError):
    """The configuration file or a configuration value is invalid."""


class CaptureError(VisionTrackError):
    """A video source could not be opened or stopped delivering frames."""


class FrameError(VisionTrackError):
    """A frame is unusable (wrong type, empty, unsupported layout, unexpected size)."""


class DetectorUnavailableError(VisionTrackError):
    """An optional detector backend was requested but cannot be loaded."""


class ReplayError(VisionTrackError):
    """A replay baseline is missing, unreadable, malformed, or invalid."""
