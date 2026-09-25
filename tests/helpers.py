"""Synthetic-frame helpers shared by the test suite. No physical camera is required anywhere."""

from __future__ import annotations

import numpy as np


def blank_frame(width: int = 200, height: int = 150, value: int = 40) -> np.ndarray:
    """A flat BGR frame (the "empty background")."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def frame_with_rect(
    x: int, y: int, w: int = 20, h: int = 20, width: int = 200, height: int = 150, bg: int = 40, fg: int = 220
) -> np.ndarray:
    """A background frame with one filled rectangle (the "moving object")."""
    frame = blank_frame(width, height, bg)
    frame[max(0, y) : y + h, max(0, x) : x + w] = fg
    return frame


def moving_rectangle_sequence(
    n: int, start_x: int = 0, step: int = 4, y: int = 60, w: int = 20, h: int = 20, width: int = 200, height: int = 150
) -> list[np.ndarray]:
    """``n`` frames of a rectangle moving left-to-right at a constant pixel/frame step."""
    return [
        frame_with_rect(start_x + i * step, y, w, h, width, height)
        for i in range(n)
    ]


def noisy_frame(width: int = 200, height: int = 150, seed: int = 0, base: int = 40, sigma: float = 6.0) -> np.ndarray:
    """A background frame with per-pixel Gaussian noise (deterministic via ``seed``)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=(height, width, 3))
    frame = np.clip(base + noise, 0, 255).astype(np.uint8)
    return frame
