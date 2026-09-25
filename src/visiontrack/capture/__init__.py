"""Frame source implementations: camera, video file, image sequence."""

from .base import Frame, FrameSource
from .camera import CameraSource
from .video import ImageSequenceSource, VideoFileSource


def open_source(uri: str, config=None):
    """Build the right :class:`FrameSource` for ``uri``.

    ``uri`` is a webcam index (``"0"``), a video file path, an existing directory (image
    sequence), or anything else (passed through as a camera/stream URI, e.g. RTSP/HTTP).
    """
    import os

    if uri.isdigit():
        return CameraSource(int(uri), config)
    if os.path.isdir(uri):
        return ImageSequenceSource(uri, config)
    if os.path.isfile(uri):
        return VideoFileSource(uri, config)
    return CameraSource(uri, config)  # network stream URL etc.


__all__ = [
    "CameraSource",
    "Frame",
    "FrameSource",
    "ImageSequenceSource",
    "VideoFileSource",
    "open_source",
]
