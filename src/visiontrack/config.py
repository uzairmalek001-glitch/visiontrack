"""Configuration schema, loading and validation.

The schema is a tree of dataclasses. :func:`config_from_dict` converts a plain mapping (parsed
YAML) into that tree and validates every value, producing errors such as::

    motion.min_area: must be >= 0 (got -5)
    zones.entrance: exactly one of 'polygon' or 'rect' is required

Unknown keys are rejected (with a "did you mean" hint) so typos never fail silently.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
import types
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml

from .errors import ConfigError
from .events.models import EventType
from .vision.geometry import is_simple_polygon, polygon_area


def cfield(default: Any = MISSING, *, factory: Any = MISSING, **meta: Any) -> Any:
    """``dataclasses.field`` with validation metadata (ge/gt/le/lt/choices/odd_or_zero/...)."""
    if factory is not MISSING:
        return field(default_factory=factory, metadata=meta)
    return field(default=default, metadata=meta)


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@dataclass
class SourceConfig:
    """Video input."""

    uri: Optional[str] = cfield(None, coerce_str=True)  # camera index, file, directory, URL
    capture_width: Optional[int] = cfield(None, ge=1)  # requested camera resolution (best effort)
    capture_height: Optional[int] = cfield(None, ge=1)
    capture_fps: Optional[float] = cfield(None, gt=0)  # requested camera frame rate
    processing_fps: Optional[float] = cfield(None, gt=0)  # decimate to at most this rate
    assumed_fps: float = cfield(30.0, gt=0)  # used when a source reports no frame rate
    max_frames: Optional[int] = cfield(None, ge=1)  # stop after this many frames read
    max_read_failures: int = cfield(5, ge=1)  # consecutive read failures tolerated


@dataclass
class PreprocessingConfig:
    """Deterministic preprocessing. Zone/line coordinates refer to the *processed* frame."""

    resize_width: Optional[int] = cfield(None, ge=1)
    resize_height: Optional[int] = cfield(None, ge=1)
    gaussian_blur_kernel: int = cfield(5, odd_or_zero=True)  # 0 disables
    denoise_kernel: int = cfield(0, odd_or_zero=True)  # median blur, 0 disables


@dataclass
class MorphologyConfig:
    """Morphological cleanup of the motion mask (all-zero iterations disables it)."""

    kernel_shape: str = cfield("ellipse", choices=("ellipse", "rect", "cross"))
    kernel_size: int = cfield(3, ge=1, odd=True)
    open_iterations: int = cfield(1, ge=0)
    close_iterations: int = cfield(0, ge=0)
    dilate_iterations: int = cfield(2, ge=0)


@dataclass
class MotionConfig:
    """Classical motion detection."""

    method: str = cfield("running_average", choices=("frame_diff", "running_average", "mog2"))
    threshold: int = cfield(25, ge=1, le=255)
    threshold_type: str = cfield("binary", choices=("binary", "otsu"))
    min_area: int = cfield(500, ge=0)
    max_area: Optional[int] = cfield(None, ge=1)
    background_alpha: float = cfield(0.05, gt=0, le=1)  # running_average learning weight
    history: int = cfield(500, ge=1)  # mog2
    var_threshold: float = cfield(16.0, gt=0)  # mog2
    detect_shadows: bool = False  # mog2
    learning_rate: Optional[float] = cfield(None, ge=0, le=1)  # mog2; None = automatic
    warmup_frames: int = cfield(5, ge=0)
    centroid_mode: str = cfield("bbox_center", choices=("bbox_center", "moments"))
    include_zones: list[str] = cfield(factory=list)  # only detect motion inside these zones
    exclude_zones: list[str] = cfield(factory=list)  # ignore motion inside these zones
    morphology: MorphologyConfig = cfield(factory=MorphologyConfig)

    def validate(self, path: str) -> None:
        if self.max_area is not None and self.max_area <= self.min_area:
            raise ConfigError(f"{path}.max_area: must be greater than min_area ({self.min_area})")


@dataclass
class TrackingConfig:
    """Centroid tracker."""

    max_disappeared: int = cfield(15, ge=0)  # frames a track may go unmatched before removal
    max_distance: float = cfield(80.0, gt=0)  # max centroid jump (px) for an association
    minimum_track_area: int = cfield(0, ge=0)  # ignore detections smaller than this
    velocity_smoothing: float = cfield(0.5, gt=0, le=1)  # EMA weight of the newest sample
    max_tracks: int = cfield(256, ge=1)  # safety cap; overflow is counted and logged
    input_sources: list[str] = cfield(factory=lambda: ["motion"])

    def validate(self, path: str) -> None:
        if not self.input_sources:
            raise ConfigError(f"{path}.input_sources: must list at least one of 'motion', 'ml'")
        for i, name in enumerate(self.input_sources):
            if name not in ("motion", "ml"):
                raise ConfigError(f"{path}.input_sources[{i}]: must be 'motion' or 'ml' (got {name!r})")


@dataclass
class ZoneConfig:
    """A polygonal (``polygon``) or rectangular (``rect`` = [x, y, w, h]) region of interest."""

    polygon: Optional[list[list[float]]] = None
    rect: Optional[list[float]] = None
    mode: str = cfield("centroid", choices=("centroid", "bbox"))  # membership test
    dwell_seconds: Optional[float] = cfield(None, gt=0)  # overrides events.dwell.threshold_s

    def validate(self, path: str) -> None:
        if (self.polygon is None) == (self.rect is None):
            raise ConfigError(f"{path}: exactly one of 'polygon' or 'rect' is required")
        if self.rect is not None:
            if len(self.rect) != 4:
                raise ConfigError(f"{path}.rect: expected [x, y, width, height]")
            if self.rect[2] <= 0 or self.rect[3] <= 0:
                raise ConfigError(f"{path}.rect: width and height must be > 0 (got {self.rect})")
            return
        assert self.polygon is not None
        if len(self.polygon) < 3:
            raise ConfigError(f"{path}.polygon: needs at least 3 points (got {len(self.polygon)})")
        for i, pt in enumerate(self.polygon):
            if len(pt) != 2:
                raise ConfigError(f"{path}.polygon[{i}]: expected [x, y] (got {pt})")
        if polygon_area(self.polygon) <= 0:
            raise ConfigError(f"{path}.polygon: has zero area (points are collinear or identical)")
        if not is_simple_polygon(self.polygon):
            raise ConfigError(f"{path}.polygon: edges intersect each other (self-intersecting polygon)")


@dataclass
class LineConfig:
    """A virtual line. ``direction`` selects which crossings are reported.

    ``any`` reports both ways; ``positive``/``negative`` report only crossings *into* the
    right-hand (positive) or left-hand (negative) side of the directed line ``start -> end`` as
    seen on screen.
    """

    start: list[float] = cfield(factory=list)
    end: list[float] = cfield(factory=list)
    direction: str = cfield("any", choices=("any", "positive", "negative"))

    def validate(self, path: str) -> None:
        for name in ("start", "end"):
            value = getattr(self, name)
            if len(value) != 2:
                raise ConfigError(f"{path}.{name}: expected [x, y] (got {value})")
        if list(self.start) == list(self.end):
            raise ConfigError(f"{path}: start and end must differ")


@dataclass
class MotionRuleConfig:
    enabled: bool = True
    start_frames: int = cfield(3, ge=1)  # consecutive motion frames before MOTION_STARTED
    stop_frames: int = cfield(15, ge=1)  # consecutive still frames before MOTION_STOPPED
    active_interval_s: float = cfield(2.0, ge=0)  # MOTION_ACTIVE period; 0 disables


@dataclass
class ZoneTransitionRuleConfig:
    enabled: bool = True
    enter_frames: int = cfield(2, ge=1)  # debounce: consecutive frames inside before ENTERED
    exit_frames: int = cfield(2, ge=1)  # debounce: consecutive frames outside before LEFT


@dataclass
class DwellRuleConfig:
    enabled: bool = True
    threshold_s: float = cfield(3.0, gt=0)  # time inside a zone before OBJECT_DWELL_STARTED


@dataclass
class LineRuleConfig:
    enabled: bool = True
    hysteresis_px: float = cfield(3.0, ge=0)  # dead band around the line to ignore jitter


@dataclass
class VelocityRuleConfig:
    enabled: bool = False
    threshold_px_s: float = cfield(500.0, gt=0)  # pixels/second, NOT a physical speed
    min_observations: int = cfield(3, ge=2)  # matched frames before velocity is trusted


@dataclass
class ZoneActivityRuleConfig:
    enabled: bool = True
    inactive_after_s: float = cfield(3.0, ge=0)  # empty time before ZONE_DEACTIVATED


@dataclass
class EventsConfig:
    """Event rules and cooldowns."""

    motion: MotionRuleConfig = cfield(factory=MotionRuleConfig)
    zone_transitions: ZoneTransitionRuleConfig = cfield(factory=ZoneTransitionRuleConfig)
    dwell: DwellRuleConfig = cfield(factory=DwellRuleConfig)
    line_crossing: LineRuleConfig = cfield(factory=LineRuleConfig)
    velocity: VelocityRuleConfig = cfield(factory=VelocityRuleConfig)
    zone_activity: ZoneActivityRuleConfig = cfield(factory=ZoneActivityRuleConfig)
    # Seconds during which a repeat of the same (type, track, zone/line) is suppressed.
    cooldowns: dict[str, float] = cfield(
        factory=lambda: {"LINE_CROSSED": 1.0, "VELOCITY_EXCEEDED": 2.0}, merge_default=True, ge=0
    )

    def validate(self, path: str) -> None:
        valid = {e.value for e in EventType}
        for name in self.cooldowns:
            if name not in valid:
                hint = difflib.get_close_matches(name, valid, n=1)
                suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
                raise ConfigError(f"{path}.cooldowns.{name}: unknown event type{suffix}")


@dataclass
class DisplayConfig:
    """Optional OpenCV window. Off by default; the engine runs headless."""

    enabled: bool = False
    window_name: str = "VisionTrack"
    show_mask: bool = False
    show_zones: bool = True
    show_tracks: bool = True
    show_velocity: bool = True
    event_ttl_s: float = cfield(3.0, gt=0)  # how long events stay on the overlay (stream time)
    max_events_shown: int = cfield(6, ge=0)
    wait_ms: int = cfield(1, ge=1)  # cv2.waitKey delay


@dataclass
class LoggingConfig:
    """Human-readable logs and JSONL event output."""

    level: str = cfield("INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    console_events: bool = True  # log every event through the standard logger
    jsonl_path: Optional[str] = None
    wall_clock_timestamps: bool = False  # adds 'wall_time' to events (non-reproducible)
    stats_interval_s: Optional[float] = cfield(None, gt=0)  # periodic performance log


@dataclass
class MLConfig:
    """Optional ML detector. Disabled by default; the core never imports ML libraries."""

    enabled: bool = False
    backend: Optional[str] = None  # 'package.module:ClassName' implementing BaseDetector
    model_path: Optional[str] = None
    confidence_threshold: float = cfield(0.5, ge=0, le=1)
    labels: list[str] = cfield(factory=list)  # keep only these labels (empty = all)
    required: bool = False  # True: fail if unavailable; False: warn and continue without ML
    options: dict[str, Any] = cfield(factory=dict)

    def validate(self, path: str) -> None:
        if self.enabled and not self.backend:
            raise ConfigError(f"{path}.backend: required when ml.enabled is true")


@dataclass
class PipelineConfig:
    max_consecutive_bad_frames: int = cfield(10, ge=1)


@dataclass
class VisionTrackConfig:
    """Root configuration object."""

    source: SourceConfig = cfield(factory=SourceConfig)
    preprocessing: PreprocessingConfig = cfield(factory=PreprocessingConfig)
    motion: MotionConfig = cfield(factory=MotionConfig)
    tracking: TrackingConfig = cfield(factory=TrackingConfig)
    zones: dict[str, ZoneConfig] = cfield(factory=dict)
    lines: dict[str, LineConfig] = cfield(factory=dict)
    events: EventsConfig = cfield(factory=EventsConfig)
    display: DisplayConfig = cfield(factory=DisplayConfig)
    logging: LoggingConfig = cfield(factory=LoggingConfig)
    ml: MLConfig = cfield(factory=MLConfig)
    pipeline: PipelineConfig = cfield(factory=PipelineConfig)

    def validate(self, path: str) -> None:
        for field_name in ("include_zones", "exclude_zones"):
            for name in getattr(self.motion, field_name):
                if name not in self.zones:
                    hint = difflib.get_close_matches(name, list(self.zones), n=1)
                    suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
                    raise ConfigError(f"motion.{field_name}: unknown zone {name!r}{suffix}")
        overlap = set(self.motion.include_zones) & set(self.motion.exclude_zones)
        if overlap:
            raise ConfigError(f"motion: zones both included and excluded: {sorted(overlap)}")
        if "ml" in self.tracking.input_sources and not self.ml.enabled:
            raise ConfigError("tracking.input_sources: 'ml' requires ml.enabled: true")
        for name in list(self.zones) + list(self.lines):
            if not name or not name.strip():
                raise ConfigError("zone and line names must be non-empty")
        clash = set(self.zones) & set(self.lines)
        if clash:
            raise ConfigError(f"zones and lines must use distinct names; shared: {sorted(clash)}")


# --------------------------------------------------------------------------------------
# Conversion / validation machinery
# --------------------------------------------------------------------------------------

_NONE_TYPE = type(None)


def _fmt(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _check_constraints(value: Any, meta: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if "ge" in meta and value < meta["ge"]:
        raise ConfigError(f"{path}: must be >= {meta['ge']} (got {_fmt(value)})")
    if "gt" in meta and value <= meta["gt"]:
        raise ConfigError(f"{path}: must be > {meta['gt']} (got {_fmt(value)})")
    if "le" in meta and value > meta["le"]:
        raise ConfigError(f"{path}: must be <= {meta['le']} (got {_fmt(value)})")
    if "lt" in meta and value >= meta["lt"]:
        raise ConfigError(f"{path}: must be < {meta['lt']} (got {_fmt(value)})")
    if meta.get("odd") and int(value) % 2 == 0:
        raise ConfigError(f"{path}: must be an odd number (got {_fmt(value)})")
    if meta.get("odd_or_zero") and not (value == 0 or (value >= 3 and int(value) % 2 == 1)):
        raise ConfigError(f"{path}: must be 0 (disabled) or an odd number >= 3 (got {_fmt(value)})")


def _convert(value: Any, tp: Any, path: str, meta: Any) -> Any:
    origin = get_origin(tp)

    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if value is None:
            if _NONE_TYPE in args:
                return None
            raise ConfigError(f"{path}: value is required")
        inner = [a for a in args if a is not _NONE_TYPE]
        return _convert(value, inner[0], path, meta)

    if tp is Any:
        return value

    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true or false (got {_fmt(value)})")
        return value

    if tp is int:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer (got {_fmt(value)})")
        _check_constraints(value, meta, path)
        return value

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number (got {_fmt(value)})")
        if not math.isfinite(value):
            raise ConfigError(f"{path}: must be a finite number (got {value})")
        _check_constraints(value, meta, path)
        return float(value)

    if tp is str:
        if meta.get("coerce_str") and isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string (got {_fmt(value)})")
        choices = meta.get("choices")
        if choices and value not in choices:
            hint = difflib.get_close_matches(value, choices, n=1)
            suffix = f"; did you mean {hint[0]!r}?" if hint else ""
            raise ConfigError(f"{path}: must be one of {list(choices)} (got {_fmt(value)}){suffix}")
        return value

    if origin is list:
        (item_tp,) = get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list (got {_fmt(value)})")
        return [_convert(v, item_tp, f"{path}[{i}]", {}) for i, v in enumerate(value)]

    if origin is dict:
        key_tp, val_tp = get_args(tp)
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping (got {_fmt(value)})")
        result = {}
        for k, v in value.items():
            key = str(k)
            result[key] = _convert(v, val_tp, f"{path}.{key}", meta)
        return result

    raise ConfigError(f"{path}: unsupported schema type {tp!r}")  # pragma: no cover


def _build(cls: type, data: Any, path: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping (got {_fmt(data)})")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    for key in data:
        if key not in fields:
            hint = difflib.get_close_matches(str(key), list(fields), n=1)
            suffix = f" (did you mean {hint[0]!r}?)" if hint else f" (valid keys: {sorted(fields)})"
            where = path or "config"
            raise ConfigError(f"{where}: unknown key {key!r}{suffix}")

    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            continue
        sub_path = f"{path}.{name}" if path else name
        value = _convert(data[name], hints[name], sub_path, f.metadata)
        if f.metadata.get("merge_default") and f.default_factory is not MISSING:  # type: ignore[misc]
            value = {**f.default_factory(), **value}  # type: ignore[misc]
        kwargs[name] = value

    obj = cls(**kwargs)
    validator = getattr(obj, "validate", None)
    if callable(validator):
        validator(path)
    return obj


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def config_from_dict(data: dict[str, Any] | None) -> VisionTrackConfig:
    """Build and validate a configuration from a mapping.

    Raises:
        ConfigError: with a human-readable, path-qualified message.
    """
    return _build(VisionTrackConfig, data, "")


def default_config() -> VisionTrackConfig:
    """A fully valid configuration made of defaults only."""
    return config_from_dict({})


def load_config(path: str | Path) -> VisionTrackConfig:
    """Load and validate a YAML configuration file.

    Raises:
        ConfigError: if the file is missing, unreadable, not valid YAML, or fails validation.
    """
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"config file not found: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(f"{file}: invalid YAML{where}: {problem}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {file}: {exc}") from exc
    if raw is not None and not isinstance(raw, dict):
        raise ConfigError(f"{file}: top level must be a mapping of sections (got {type(raw).__name__})")
    try:
        return config_from_dict(raw)
    except ConfigError as exc:
        raise ConfigError(f"{file}: {exc}") from exc


def config_to_dict(config: VisionTrackConfig) -> dict[str, Any]:
    """Plain-dict form of a configuration (round-trips through :func:`config_from_dict`)."""
    return dataclasses.asdict(config)


def dump_config(config: VisionTrackConfig) -> str:
    """Effective configuration as YAML text."""
    return yaml.safe_dump(config_to_dict(config), sort_keys=False, default_flow_style=None)
