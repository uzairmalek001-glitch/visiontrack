import tempfile
import unittest
from pathlib import Path

from visiontrack.config import ConfigError, config_from_dict, default_config, load_config


class DefaultsTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = default_config()
        self.assertEqual(config.motion.method, "running_average")
        self.assertFalse(config.ml.enabled)
        self.assertFalse(config.display.enabled)


class ValidationTests(unittest.TestCase):
    def test_unknown_top_level_key_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            config_from_dict({"totally_unknown_section": {}})
        self.assertIn("unknown key", str(ctx.exception))

    def test_typo_key_gets_suggestion(self):
        with self.assertRaises(ConfigError) as ctx:
            config_from_dict({"moton": {}})
        self.assertIn("did you mean", str(ctx.exception))

    def test_negative_min_area_rejected(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"motion": {"min_area": -5}})

    def test_max_area_must_exceed_min_area(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"motion": {"min_area": 500, "max_area": 100}})

    def test_even_blur_kernel_rejected(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"preprocessing": {"gaussian_blur_kernel": 4}})

    def test_zero_blur_kernel_allowed_disables_it(self):
        config = config_from_dict({"preprocessing": {"gaussian_blur_kernel": 0}})
        self.assertEqual(config.preprocessing.gaussian_blur_kernel, 0)

    def test_invalid_method_choice_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            config_from_dict({"motion": {"method": "not_a_method"}})
        self.assertIn("must be one of", str(ctx.exception))

    def test_zone_needs_polygon_or_rect(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"zones": {"a": {}}})

    def test_zone_cannot_have_both_polygon_and_rect(self):
        with self.assertRaises(ConfigError):
            config_from_dict(
                {"zones": {"a": {"polygon": [[0, 0], [1, 0], [1, 1]], "rect": [0, 0, 1, 1]}}}
            )

    def test_zone_polygon_needs_three_points(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"zones": {"a": {"polygon": [[0, 0], [1, 1]]}}})

    def test_zone_polygon_zero_area_rejected(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"zones": {"a": {"polygon": [[0, 0], [10, 0], [20, 0]]}}})

    def test_zone_self_intersecting_polygon_rejected(self):
        with self.assertRaises(ConfigError):
            config_from_dict(
                {"zones": {"a": {"polygon": [[0, 0], [10, 10], [10, 0], [0, 10]]}}}
            )

    def test_valid_rect_zone_accepted(self):
        config = config_from_dict({"zones": {"a": {"rect": [0, 0, 10, 10]}}})
        self.assertIn("a", config.zones)

    def test_zone_rect_bad_length_rejected(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"zones": {"a": {"rect": [0, 0, 10]}}})

    def test_line_needs_distinct_endpoints(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"lines": {"l": {"start": [0, 0], "end": [0, 0]}}})

    def test_zone_and_line_cannot_share_a_name(self):
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "zones": {"x": {"rect": [0, 0, 10, 10]}},
                    "lines": {"x": {"start": [0, 0], "end": [10, 10]}},
                }
            )

    def test_motion_include_zone_must_exist(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"motion": {"include_zones": ["missing"]}})

    def test_motion_zone_cannot_be_both_included_and_excluded(self):
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "zones": {"a": {"rect": [0, 0, 10, 10]}},
                    "motion": {"include_zones": ["a"], "exclude_zones": ["a"]},
                }
            )

    def test_unknown_cooldown_event_type_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            config_from_dict({"events": {"cooldowns": {"NOT_A_REAL_EVENT": 1.0}}})
        self.assertIn("unknown event type", str(ctx.exception))

    def test_ml_enabled_requires_backend(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"ml": {"enabled": True}})

    def test_ml_input_source_requires_ml_enabled(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"tracking": {"input_sources": ["motion", "ml"]}})

    def test_ml_enabled_with_backend_is_valid(self):
        config = config_from_dict({"ml": {"enabled": True, "backend": "pkg.mod:Cls"}})
        self.assertEqual(config.ml.backend, "pkg.mod:Cls")

    def test_bool_is_not_accepted_as_int(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"motion": {"min_area": True}})

    def test_default_cooldowns_are_preserved_when_adding_one(self):
        config = config_from_dict({"events": {"cooldowns": {"LINE_CROSSED": 5.0}}})
        self.assertEqual(config.events.cooldowns["LINE_CROSSED"], 5.0)
        self.assertIn("VELOCITY_EXCEEDED", config.events.cooldowns)


class FileLoadingTests(unittest.TestCase):
    def test_missing_file_raises_config_error(self):
        with self.assertRaises(ConfigError):
            load_config("/no/such/file.yaml")

    def test_non_mapping_top_level_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.yaml"
            path.write_text("- 1\n- 2\n")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_invalid_yaml_raises_config_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.yaml"
            path.write_text("motion: [unterminated\n")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_valid_file_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ok.yaml"
            path.write_text("motion:\n  min_area: 123\n")
            config = load_config(path)
            self.assertEqual(config.motion.min_area, 123)

    def test_example_config_is_valid(self):
        example = Path(__file__).resolve().parents[1] / "config" / "example.yaml"
        config = load_config(example)
        self.assertIn("entrance", config.zones)
        self.assertIn("midline", config.lines)


if __name__ == "__main__":
    unittest.main()
