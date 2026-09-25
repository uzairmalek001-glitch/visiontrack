"""VisionTrack command-line interface."""

from __future__ import annotations

import argparse
import sys

from .capture import open_source
from .config import ConfigError, VisionTrackConfig, config_to_dict, default_config, load_config
from .errors import VisionTrackError
from .output.jsonl import JsonlEventWriter
from .output.logger import configure_logging, console_event_sink
from .pipeline import Pipeline
from .replay import compare_events, replay


def _load(config_path: str | None) -> VisionTrackConfig:
    return load_config(config_path) if config_path else default_config()


def _apply_cli_overrides(config: VisionTrackConfig, args: argparse.Namespace) -> None:
    if getattr(args, "display", False):
        config.display.enabled = True
    if getattr(args, "headless", False):
        config.display.enabled = False
    if getattr(args, "jsonl", None):
        config.logging.jsonl_path = args.jsonl


def cmd_run(args: argparse.Namespace) -> int:
    config = _load(args.config)
    _apply_cli_overrides(config, args)
    logger = configure_logging(config.logging)

    pipeline = Pipeline(config=config, source_name=args.source, keep_color=config.display.enabled)
    if config.logging.console_events:
        pipeline.add_event_sink(console_event_sink(logger))

    jsonl_writer = None
    if config.logging.jsonl_path:
        jsonl_writer = JsonlEventWriter(config.logging.jsonl_path)
        jsonl_writer.open()
        pipeline.add_event_sink(jsonl_writer)

    try:
        source = open_source(args.source, config.source)
        metrics = pipeline.run(source)
    except VisionTrackError as exc:
        logger.error(str(exc))
        return 1
    finally:
        if jsonl_writer is not None:
            jsonl_writer.close()

    logger.info("stopped: %s", metrics.snapshot())
    return 0


def cmd_inspect_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.path)
    except ConfigError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 1
    import yaml

    print(yaml.safe_dump(config_to_dict(config), sort_keys=False, default_flow_style=None))
    print("Configuration is valid.", file=sys.stderr)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    config = _load(args.config)
    _apply_cli_overrides(config, args)
    logger = configure_logging(config.logging)
    try:
        events, metrics = replay(args.source, config, jsonl_out=args.jsonl)
    except VisionTrackError as exc:
        logger.error(str(exc))
        return 1

    logger.info("replay produced %d events over %d frames", len(events), metrics.frames_processed)
    if args.compare:
        diffs = compare_events(args.compare, events)
        if diffs:
            logger.error("replay diverged from baseline at %d position(s):", len(diffs))
            for d in diffs[:10]:
                logger.error("  #%d expected=%s actual=%s", d.index, d.expected, d.actual)
            return 1
        logger.info("replay matches baseline exactly (%s)", args.compare)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visiontrack", description="Real-time CV event detection and tracking.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Process a live camera or video source.")
    run.add_argument("--source", required=True, help="Camera index (e.g. 0), video file, or image directory.")
    run.add_argument("--config", help="Path to a YAML configuration file.")
    run.add_argument("--display", action="store_true", help="Show an OpenCV visualization window.")
    run.add_argument("--headless", action="store_true", help="Force-disable the display window.")
    run.add_argument("--jsonl", help="Path to write JSONL events to (overrides config).")
    run.set_defaults(func=cmd_run)

    inspect = sub.add_parser("inspect-config", help="Validate a config file and print the effective configuration.")
    inspect.add_argument("path", help="Path to a YAML configuration file.")
    inspect.set_defaults(func=cmd_inspect_config)

    rep = sub.add_parser("replay", help="Deterministically replay a video/image-sequence source.")
    rep.add_argument("source", help="Video file or image directory.")
    rep.add_argument("--config", help="Path to a YAML configuration file.")
    rep.add_argument("--jsonl", help="Write the replayed events to this JSONL file.")
    rep.add_argument("--compare", help="Compare against a previously recorded JSONL baseline.")
    rep.add_argument("--display", action="store_true", help="Show an OpenCV visualization window.")
    rep.add_argument("--headless", action="store_true", help="Force-disable the display window.")
    rep.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except VisionTrackError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
