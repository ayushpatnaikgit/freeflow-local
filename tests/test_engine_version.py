"""Tests for the engine version reported by /health (used by apps to offer engine updates).

Run: uv run python -m unittest discover tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


class EngineVersionTest(unittest.TestCase):
    def test_release_install_reports_its_tag(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "VERSION"
            f.write_text("v0.3.6\n")
            self.assertEqual(server.engine_version(f), "v0.3.6")

    def test_git_checkout_install_reports_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(server.engine_version(Path(d) / "VERSION"))
            empty = Path(d) / "EMPTY"
            empty.write_text("\n")
            self.assertIsNone(server.engine_version(empty))


if __name__ == "__main__":
    unittest.main()
