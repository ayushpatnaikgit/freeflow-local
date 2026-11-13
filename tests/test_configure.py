"""`freeflowctl configure` writes each app's settings file: FreeFlow needs a placeholder API key, TellWell has none."""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FREEFLOWCTL = Path(__file__).resolve().parents[1] / "freeflowctl"
URL = "http://127.0.0.1:8081/v1"


def configure_script() -> str:
    match = re.search(r"\n  configure\).*?<<'EOF'\n(.*?)\nEOF\n", FREEFLOWCTL.read_text(), re.S)
    if match is None:
        raise AssertionError("freeflowctl configure no longer embeds its settings writer")
    return match.group(1)


def configure(app: str, existing: dict | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".settings"
        if existing is not None:
            path.write_text(json.dumps(existing))
        subprocess.run([sys.executable, "-c", configure_script(), str(path), URL, app],
                       check=True, capture_output=True)
        return json.loads(path.read_text())


class ConfigureTests(unittest.TestCase):
    def test_tellwell_gets_no_api_key(self):
        settings = configure("TellWell")
        self.assertEqual(settings["api_base_url"], URL)
        self.assertNotIn("groq_api_key", settings)

    def test_tellwell_drops_the_old_placeholder_but_not_a_real_key(self):
        self.assertNotIn("groq_api_key", configure("TellWell", {"groq_api_key": "local"}))
        self.assertEqual(configure("TellWell", {"groq_api_key": "synthetic-real-key"})["groq_api_key"],
                         "synthetic-real-key")

    def test_freeflow_still_gets_a_placeholder_key(self):
        self.assertEqual(configure("FreeFlow")["groq_api_key"], "local")
        self.assertEqual(configure("FreeFlow", {"groq_api_key": "synthetic-real-key"})["groq_api_key"],
                         "synthetic-real-key")


if __name__ == "__main__":
    unittest.main()
