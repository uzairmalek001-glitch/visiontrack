# VisionTrack

**Real-time computer-vision event detection and tracking engine.**

VisionTrack watches a camera or video file with classical OpenCV — no machine-learning model
required — and turns raw motion into structured, debounced events: something entered a zone,
lingered too long, crossed a line, or moved unusually fast. It's built for people who want a
motion/zone/line event pipeline they can actually reason about, extend, and run without a GPU.

> **Status:** V1, functional and tested (137 automated tests, see [Testing](#testing)). This is
> a solid foundation, not a finished commercial product — read [Limitations](#limitations)
> before you point it at anything that matters.

---

## 1. Project overview

VisionTrack processes a video stream frame by frame through a fixed pipeline — capture,
preprocess, detect motion, check regions of interest, track objects, and emit events — and
writes the result as structured JSON Lines, human-readable logs, and/or an optional on-screen
overlay. Everything runs locally; nothing is uploaded anywhere.

The core design decision is **deterministic-first**: classical OpenCV (background subtraction,
contours, geometry) does all the real work, and it needs no model file, no GPU, and no internet
connection. An ML object detector is a clean, entirely optional plug-in — the engine is
identical with or without one, and V1 ships no bundled model.

## 2. Architecture

```
 Video Source (camera / file / image sequence)
        │
        ▼
 Frame Capture  ──────────────────────────────────────────►  Frame{image, frame_number, timestamp}
        │
        ▼
 Preprocessing (resize → grayscale → denoise → blur)
        │
        ▼
 Motion Detection (frame-diff / running-average / MOG2 → contours) ──► Detection[] (source="motion")
        │                                                                    ▲
        │                                                    Optional ML detector (source="ml")
        ▼                                                                    │
 Region-of-Interest membership  ◄──────────────── unified Detection model ──┘
        │
        ▼
 Tracking (deterministic centroid tracker) ──► Track[] (id, centroid, velocity, zones...)
        │
        ▼
 State Management (zone occupancy debounce, dwell timers)
        │
        ▼
 Event Engine (rules → cooldown/debounce → stamped Event)
        │
        ▼
 Dispatch: console logger · JSONL writer · OpenCV overlay
```

Every stage is a small, independently-testable module (see [Testing](#testing)). The `Detection`
type is the seam between "how was this object found" (motion contour or ML model) and
"everything downstream" (tracking, zones, events) — trackers and rules never know or care which
detector produced a box.

```
src/visiontrack/
├── capture/     FrameSource: camera, video file, image sequence (+ open_source() dispatcher)
├── vision/      preprocessing, motion detection, contour extraction, geometry helpers
├── tracking/    BaseTracker + deterministic CentroidTracker, Track state
├── regions/     Region (polygon/rect) geometry + debounced ZoneOccupancyTracker
├── events/      EventType/Event models, per-behavior rule state machines, EventEngine
├── detectors/   BaseDetector interface + optional dynamic ML backend loader
├── output/      console logger, JSONL writer, OpenCV overlay
├── config.py    dataclass schema, YAML loading, validation with helpful errors
├── pipeline.py  wires every stage together; Pipeline.process_frame() is a pure, testable step
├── replay.py    deterministic replay + JSONL baseline comparison
└── cli.py       `visiontrack run / inspect-config / replay`
```

## 3. Features

- **Motion detection** — frame differencing, running-average background, or MOG2, with
  configurable thresholding and morphological cleanup (open/close/dilate).
- **Zones** — polygonal or rectangular regions of interest, with debounced enter/leave/dwell
  events and per-zone dwell overrides.
- **Line crossing** — a virtual line with configurable direction filtering (either way, or only
  one side) and a hysteresis dead-band so jitter on the line doesn't double-count.
- **Deterministic centroid tracking** — stable IDs, temporary-disappearance tolerance, smoothed
  pixel-per-second velocity and coarse direction.
- **Event engine** — eleven event types (see [Event types](#event-types)) produced from state
  *transitions*, not raw per-frame noise, with per-event-type cooldowns.
- **Optional ML detector slot** — a `BaseDetector` interface and dynamic backend loader; the
  deterministic engine works identically with it absent.
- **JSONL + console logging**, an optional OpenCV overlay, and a headless mode for servers/CI.
- **Deterministic replay** — `visiontrack replay` reproduces the exact same event sequence for
  the same input file and config, and can diff against a recorded baseline for regression tests.
- **Config validation with useful errors** ("did you mean...?", range checks, self-intersecting
  polygon detection) instead of cryptic tracebacks.

## 4. Installation

Requires Python 3.11+.

```bash
git clone <this-repo>
cd visiontrack
pip install -e .
# or, for running the test suite too:
pip install -e ".[dev]"
```

On a headless server, you can swap `opencv-python` for `opencv-python-headless` in
`pyproject.toml` — VisionTrack never requires a display (see `--headless` below).

## 5. Quick start

```bash
# Validate a config and see the fully-resolved settings:
visiontrack inspect-config config/example.yaml

# Process a video file, printing events to the console:
visiontrack run --source examples/test_video.mp4 --config config/example.yaml --headless

# Same, but also write structured events to a file:
visiontrack run --source examples/test_video.mp4 --config config/example.yaml \
    --headless --jsonl out/events.jsonl

# Live webcam, with an on-screen overlay:
visiontrack run --source 0 --display
```

`examples/test_video.mp4` is a small synthetic clip (a rectangle sliding across frame) generated
for this repo so you can try VisionTrack with no camera or external footage.
`examples/sample_events.jsonl` shows the JSONL output it produces with `config/example.yaml`.

## 6. CLI usage

```
visiontrack run --source <src> [--config PATH] [--display | --headless] [--jsonl PATH]
visiontrack inspect-config <path>
visiontrack replay <source> [--config PATH] [--jsonl PATH] [--compare BASELINE] [--display|--headless]
```

- `--source` accepts a webcam index (`0`, `1`, ...), a video file path, a directory of images
  (read in sorted filename order), or a stream URL (RTSP/HTTP — passed through to OpenCV).
- `--display` opens an OpenCV window; `--headless` force-disables it (useful in CI even if the
  config file turns display on). Neither is required — the engine defaults to headless.
- `inspect-config` validates a file and prints the fully-resolved configuration (with every
  default filled in) so you can see exactly what will run.
- `replay` is `run` without live-source concerns, plus optional baseline comparison for
  regression testing (see [Testing](#testing)).

## 7. Configuration

YAML, validated at startup with path-qualified error messages, e.g.:

```
motion.max_area: must be greater than min_area (500)
zones.entrance: exactly one of 'polygon' or 'rect' is required
moton: unknown key (did you mean 'motion'?)
```

See `config/example.yaml` for a complete, runnable example. Key sections:

```yaml
source: {processing_fps, assumed_fps, capture_width/height, max_frames, ...}
preprocessing: {resize_width/height, gaussian_blur_kernel, denoise_kernel}
motion: {method, threshold, min_area, max_area, morphology: {...}, include_zones, exclude_zones}
tracking: {max_disappeared, max_distance, minimum_track_area, velocity_smoothing, max_tracks}
zones:
  entrance:
    polygon: [[x, y], ...]        # or rect: [x, y, w, h]
    mode: centroid | bbox
    dwell_seconds: 5.0            # optional per-zone override
lines:
  midline: {start: [x, y], end: [x, y], direction: any | positive | negative}
events:
  motion: {start_frames, stop_frames, active_interval_s}
  zone_transitions: {enter_frames, exit_frames}
  dwell: {threshold_s}
  line_crossing: {hysteresis_px}
  velocity: {enabled, threshold_px_s, min_observations}
  cooldowns: {LINE_CROSSED: 1.0, VELOCITY_EXCEEDED: 2.0, ...}
display: {enabled, show_zones, show_tracks, show_velocity, show_mask}
logging: {level, console_events, jsonl_path, wall_clock_timestamps}
ml: {enabled, backend, model_path, confidence_threshold, required}
```

Zone and line coordinates refer to the **processed** frame (after `preprocessing.resize_*`), not
the source resolution.

## 8. Event types

| Event | Fires when |
|---|---|
| `MOTION_STARTED` / `MOTION_STOPPED` | Motion presence changes, debounced over N frames |
| `MOTION_ACTIVE` | Periodic heartbeat while motion continues |
| `OBJECT_ENTERED_ZONE` / `OBJECT_LEFT_ZONE` | Debounced zone membership change for a track |
| `OBJECT_DWELL_STARTED` / `OBJECT_DWELL_ENDED` | A track has been in a zone ≥ threshold seconds |
| `LINE_CROSSED` | A track's centroid crosses a configured line (with hysteresis) |
| `VELOCITY_EXCEEDED` | A track's smoothed speed passes a pixel/second threshold |
| `ZONE_ACTIVATED` / `ZONE_DEACTIVATED` | A zone gains its first occupant / has been empty ≥ threshold |

Every event carries `event_id`, `event_type`, `timestamp` (stream time, not wall clock, unless
`logging.wall_clock_timestamps` is set), `frame_number`, `source`, and, where applicable,
`track_id`, `zone`, `bbox`, and `metadata`. Per-event-type cooldowns suppress repeat firing of
the *same* (type, track, zone/line) combination within a configurable window — on top of, not
instead of, the frame-debouncing each rule already does internally.

## 9. Tracking architecture

`CentroidTracker` is a deterministic greedy nearest-centroid matcher: for each frame it computes
every (track, detection) distance under `max_distance`, and repeatedly commits the globally
smallest remaining distance, with ties broken by `(track_id, detection_index)` — so the same
input always produces the same associations. Unmatched tracks accumulate
`disappeared_frames` and are dropped after `max_disappeared` consecutive misses; unmatched
detections become new tracks (capped at `max_tracks`, with overflow counted, not silently
dropped). Velocity is an exponentially-smoothed pixel/second vector — **pixels, not physical
units**: VisionTrack does no camera calibration, so "speed" and "direction" are screen-space
only.

## 10. Testing

**137 automated tests, all passing**, none requiring a physical camera:

```bash
pytest                                          # if pytest is installed
python -m unittest discover -s tests -t . -v    # stdlib fallback, no install needed
```

| File | Count | Covers |
|---|---|---|
| `tests/test_geometry.py` | 26 | bbox/IoU, polygon membership & self-intersection, line intersection |
| `tests/test_config.py` | 30 | schema validation, typo suggestions, YAML error handling |
| `tests/test_motion.py` | 11 | all three motion methods, ROI masking, noise rejection |
| `tests/test_tracking.py` | 13 | ID stability, disappearance, velocity, overflow |
| `tests/test_regions.py` | 15 | zone membership modes, debounced enter/leave/dwell |
| `tests/test_events.py` | 21 | every rule, cooldowns, event sequencing |
| `tests/integration/test_pipeline_video.py` | 8 | full `Pipeline` on real (synthetic) video files, deterministic replay, corrupted/missing files |
| `tests/optional/test_ml_detector.py` | 9 | ML loader with **no ML framework installed** |

Unit tests use synthetic frames (`tests/helpers.py`: static background, moving rectangle,
Gaussian noise). Integration tests write and decode real `.mp4` files through OpenCV's actual
codec path. `visiontrack replay <video> --jsonl baseline.jsonl` followed by
`visiontrack replay <video> --compare baseline.jsonl` is the regression-testing workflow for
catching behavior changes after an algorithm edit.

## 11. Performance considerations

`visiontrack run` logs a metrics snapshot on exit (capture/processing FPS, latency mean/p95/max,
active/max tracks, events/minute, frames skipped/dropped/late). There is no threading or
multiprocessing in V1 — the loop is intentionally simple and easy to reason about; if the
processing FPS metric shows you falling behind your source's frame rate, first try
`preprocessing.resize_width` (smaller frames) or `source.processing_fps` (frame decimation)
before reaching for concurrency.

## 12. Privacy model

- No frame data is uploaded anywhere; no cloud API is called automatically.
- JSONL output contains structured events only (`bbox`, `centroid`, timestamps, zone/track
  IDs) — **never raw frame pixels**.
- No video is stored by default; nothing is written to disk unless you set `logging.jsonl_path`
  or explicitly wire in your own frame-saving code.
- If you enable an ML backend, inference happens wherever *your* adapter code puts it — nothing
  in VisionTrack's core sends frames off-device. Document your adapter's behavior if it does.

## 13. Optional ML architecture

```
ml:
  enabled: true
  backend: "myproject.detectors.yolo:YoloDetector"   # package.module:ClassName
  model_path: "yolov8n.onnx"
  required: false     # false: fall back to motion-only if the backend fails to load
```

`visiontrack.detectors.optional_ml.load_detector()` dynamically imports **only** the module
named in `backend` — the core (`pipeline.py`, `tracking/`, `events/`) never imports PyTorch,
ONNX Runtime, TensorFlow, or any ML library. A machine with none of those installed runs the
full deterministic engine untouched; `tests/optional/test_ml_detector.py` verifies exactly this.
No adapter ships with V1 — implement `visiontrack.detectors.base.BaseDetector.detect()` to add
one; it must tag every detection `source="ml"` with a `confidence` in `[0, 1]`.

## 14. Limitations

Read this before deploying VisionTrack anywhere that matters:

- No accuracy benchmarks. Motion-based detection has no measured precision/recall on a standard dataset and has not been formally benchmarked.
- No physical speed or distance. Velocity and speed are measured in pixels/second in the processed frame. They are not real-world units without camera calibration, which V1 does not provide.
- Not robust tracking. The centroid tracker is deterministic and testable, but it is not state-of-the-art. Heavy occlusion, fast crossing paths, and crowded scenes can cause track identities to be confused, and there is no re-identification.
- Greedy association is not globally optimal. The tracker deterministically commits the nearest available track/detection pair first. In crowded scenes, this can produce a higher-cost assignment and incorrect track identities.
- Duplicate detections are not deduplicated. If an upstream detector produces multiple identical or near-identical detections, VisionTrack can create multiple track IDs for the same apparent object.
- Classical motion detection has known failure modes. Illumination changes, shadows, camera shake, and reflections can trigger false positives, while slow-moving or low-contrast objects can be missed.
- No ML adapter ships with V1. The interface is complete and tested, but users must provide or implement their own backend for learned object detection.
- No security hardening has been demonstrated. “Local/offline by default” is a design property, not a security audit or security guarantee.
- Single-threaded and single-process. VisionTrack makes no claim of real-time performance on specific hardware. Use the built-in metrics to measure performance on your target system.

## 15. Future roadmap

- A reference ML adapter, such as ONNX Runtime + a small YOLO model, behind the existing `BaseDetector` interface while remaining fully optional.
- Multi-object re-identification across brief full occlusions.
- Native RTSP/network-stream reconnect-on-drop handling in `CameraSource`.
- Optional frame/clip recording around events, explicitly opt-in to preserve the no-video-by-default privacy stance.
- A minimal web dashboard consuming the JSONL event stream.
- Camera calibration support for real-world speed and distance units.
