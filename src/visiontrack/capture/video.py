"""Video-file and image-sequence frame sources."""

from __future__ import annotations

from pathlib import Path

import cv2

from ..config import SourceConfig
from ..errors import CaptureError
from .base import Frame, FrameSource

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class VideoFileSource(FrameSource):
    """Reads frames from a video file. End of file yields ``None`` from :meth:`read`, it never
    raises — playback ending normally is not an error.
    """

    def __init__(self, path: str, config: SourceConfig | None = None) -> None:
        self._path = path
        self._cfg = config or SourceConfig()
        self._cap: cv2.VideoCapture | None = None
        self._fps = self._cfg.assumed_fps
        self._size: tuple[int, int] | None = None
        self._frame_number = 0
        self._consecutive_failures = 0

    def open(self) -> None:
        if not Path(self._path).is_file():
            raise CaptureError(f"video file not found: {self._path}")
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"could not open video file: {self._path}")
        reported = cap.get(cv2.CAP_PROP_FPS)
        if reported and reported > 0:
            self._fps = reported
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self._size = (w, h)
        self._cap = cap
        self._frame_number = 0
        self._consecutive_failures = 0

    def read(self) -> Frame | None:
        if self._cap is None:
            raise CaptureError("read() called before open()")
        if self._cfg.max_frames is not None and self._frame_number >= self._cfg.max_frames:
            return None
        ok, image = self._cap.read()
        if not ok:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._cfg.max_read_failures:
                return None
            return self.read()
        self._consecutive_failures = 0
        frame = Frame(
            image=image,
            frame_number=self._frame_number,
            timestamp=self._frame_number / self._fps,
        )
        self._frame_number += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def name(self) -> str:
        return Path(self._path).name


class ImageSequenceSource(FrameSource):
    """Reads frames from a directory of images in sorted filename order.

    Timestamps are synthetic (``frame_number / assumed_fps``) since still images carry no
    native frame rate.
    """

    def __init__(self, directory: str, config: SourceConfig | None = None) -> None:
        self._dir = directory
        self._cfg = config or SourceConfig()
        self._files: list[Path] = []
        self._index = 0
        self._size: tuple[int, int] | None = None

    def open(self) -> None:
        directory = Path(self._dir)
        if not directory.is_dir():
            raise CaptureError(f"image sequence directory not found: {self._dir}")
        self._files = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._files:
            raise CaptureError(f"no images found in {self._dir}")
        self._index = 0

    def read(self) -> Frame | None:
        if self._cfg.max_frames is not None and self._index >= self._cfg.max_frames:
            return None
        if self._index >= len(self._files):
            return None
        path = self._files[self._index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureError(f"could not decode image: {path}")
        if self._size is None:
            self._size = (image.shape[1], image.shape[0])
        frame = Frame(
            image=image,
            frame_number=self._index,
            timestamp=self._index / self._cfg.assumed_fps,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return self._cfg.assumed_fps

    @property
    def frame_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def name(self) -> str:
        return Path(self._dir).name
