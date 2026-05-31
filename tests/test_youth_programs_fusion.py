"""Unit tests for youth program fusion helpers (stdlib unittest only)."""

from pathlib import Path
import sys
import unittest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "examples" / "youth_programs_academy"))

from fusion import fuse_programs, normalize_programs_payload, safe_parse_raw


class TestFusion(unittest.TestCase):
    def test_fuse_prefers_primary(self):
        primary = [{"name": "A", "joiningLink": "https://x.com/a", "description": "from_primary"}]
        secondary = [{"name": "A", "joiningLink": "https://x.com/a", "description": "from_secondary"}]
        out = fuse_programs(primary, secondary, prefer_primary=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["description"], "from_primary")

    def test_normalize_sources_meta(self):
        raw = {"programs": [{"name": "Z"}], "sources": ["https://x.org"]}
        n = normalize_programs_payload(raw)
        self.assertEqual(n["programs"][0]["name"], "Z")
        self.assertIn("_meta_sources", n)

    def test_safe_parse_pipe_joined_json(self):
        raw = '{"programs": []}|{"programs": [{"name": "P"}]}'
        out = safe_parse_raw(raw)
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["programs"]), 1)
        self.assertEqual(out["programs"][0]["name"], "P")

    def test_normalize_pipe_concatenated_mergeanswers(self):
        raw = '{"programs": []}|{"programs": [{"name": "Q"}]}'
        n = normalize_programs_payload(raw)
        self.assertEqual(len(n["programs"]), 1)
        self.assertEqual(n["programs"][0]["name"], "Q")


if __name__ == "__main__":
    unittest.main()
