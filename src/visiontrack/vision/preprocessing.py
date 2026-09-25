"""Deterministic frame preprocessing and mask post-processing.

Stages that are disabled in the configuration cost nothing: no function is called and no
array is allocated for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import MorphologyConfig, PreprocessingConfig
from ..errors import FrameError

_KERNEL_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


@dataclass(slots=True)
class PreprocessedFrame:
    """Result of preprocessing one frame.

    Attributes:
        gray: Single-channel ``uint8`` image used by the vision stages.
        color: Resized BGR image, only produced when a consumer (overlay) needs it.
        size: ``(width, height)`` of the processed frame; zone/line coordinates use this space.
    """

    gray: np.ndarray
    color: np.ndarray | None
    size: tuple[int, int]


def validate_image(image: object) -> np.ndarray:
    """Return ``image`` if it is a usable frame, otherwise raise :class:`FrameError`."""
    if not isinstance(image, np.ndarray):
        raise FrameError(f"frame must be a numpy array, got {type(image).__name__}")
    if image.size == 0:
        raise FrameError("frame is empty")
    if image.dtype != np.uint8:
        raise FrameError(f"unsupported frame dtype {image.dtype}; expected uint8")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        return image
    raise FrameError(f"unsupported frame layout with shape {image.shape}")


def compute_target_size(
    in_width: int, in_height: int, width: int | None, height: int | None
) -> tuple[int, int]:
    """Target ``(w, h)``: exact if both given, aspect-preserving if only one, else unchanged."""
    if width is not None and height is not None:
        return width, height
    if width is not None:
        return width, max(1, round(in_height * width / in_width))
    if height is not None:
        return max(1, round(in_width * height / in_height)), height
    return in_width, in_height


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a validated frame to single-channel grayscale (no copy if already gray)."""
    if image.ndim == 2:
        return image
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a validated frame to 3-channel BGR (no copy if already BGR)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


class Preprocessor:
    """Resize -> grayscale -> optional median denoise -> optional Gaussian blur.

    Args:
        config: Preprocessing configuration.
        keep_color: Also return the resized BGR frame (needed only for the overlay).
    """

    def __init__(self, config: PreprocessingConfig, keep_color: bool = False) -> None:
        self._cfg = config
        self._keep_color = keep_color
        self._resize_enabled = config.resize_width is not None or config.resize_height is not None
        self._locked_input: tuple[int, int] | None = None

    def process(self, image: np.ndarray) -> PreprocessedFrame:
        """Preprocess one frame.

        Raises:
            FrameError: for unusable frames, or if the frame size changes mid-stream while
                resizing is disabled (downstream ROI coordinates would silently become wrong).
        """
        image = validate_image(image)
        in_h, in_w = image.shape[:2]

        if not self._resize_enabled:
            if self._locked_input is None:
                self._locked_input = (in_w, in_h)
            elif self._locked_input != (in_w, in_h):
                raise FrameError(
                    f"frame size changed from {self._locked_input[0]}x{self._locked_input[1]} "
                    f"to {in_w}x{in_h} mid-stream; set preprocessing.resize_width/height to "
                    "normalise the size"
                )

        out_w, out_h = compute_target_size(
            in_w, in_h, self._cfg.resize_width, self._cfg.resize_height
        )
        if (out_w, out_h) != (in_w, in_h):
            interpolation = cv2.INTER_AREA if out_w * out_h < in_w * in_h else cv2.INTER_LINEAR
            image = cv2.resize(image, (out_w, out_h), interpolation=interpolation)

        gray = to_gray(image)
        if self._cfg.denoise_kernel:
            gray = cv2.medianBlur(gray, self._cfg.denoise_kernel)
        if self._cfg.gaussian_blur_kernel:
            k = self._cfg.gaussian_blur_kernel
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        color = to_bgr(image) if self._keep_color else None
        return PreprocessedFrame(gray=gray, color=color, size=(out_w, out_h))


def threshold_mask(image: np.ndarray, value: int, mode: str = "binary") -> np.ndarray:
    """Binarise ``image`` to a 0/255 mask.

    Args:
        image: Single-channel ``uint8`` image.
        value: Threshold for ``mode='binary'`` (pixels strictly greater become 255).
        mode: ``'binary'`` for a fixed threshold, ``'otsu'`` for automatic selection.
    """
    if mode == "otsu":
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
    return mask


class MorphologyFilter:
    """Opening -> closing -> dilation with a pre-built structuring element.

    Opening removes speckle noise, closing fills small holes, dilation merges fragments of one
    object (at the price of growing boxes by roughly one pixel per iteration per side).
    """

    def __init__(self, config: MorphologyConfig) -> None:
        self._cfg = config
        self._enabled = bool(
            config.open_iterations or config.close_iterations or config.dilate_iterations
        )
        size = config.kernel_size
        self._kernel = cv2.getStructuringElement(_KERNEL_SHAPES[config.kernel_shape], (size, size))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return mask
        cfg = self._cfg
        if cfg.open_iterations:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=cfg.open_iterations)
        if cfg.close_iterations:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=cfg.close_iterations)
        if cfg.dilate_iterations:
            mask = cv2.dilate(mask, self._kernel, iterations=cfg.dilate_iterations)
        return mask
