"""Tests for serving a fine-tuned adapter: the compact prompt swap and the adapter setting. Synthetic data only.

Run: uv run python -m unittest discover tests
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compact_prompt  # noqa: E402
import server  # noqa: E402

FREEFLOW_DEFAULT = ("You are a literal dictation cleanup layer. Rule one.\nRule two.\n"
                    "A standalone reply such as \"yes\" is still dictation.")
VOCAB_TAIL = ("\n\nThe following vocabulary must be treated as high-priority terms while rewriting.\n"
              "Use these spellings exactly in the output when relevant:\nZentrix, Q-Lattice")


class CompactPromptTest(unittest.TestCase):
    def test_default_prompt_is_swapped_and_vocabulary_kept(self):
        out = compact_prompt.compact_system_prompt(FREEFLOW_DEFAULT + VOCAB_TAIL)
        self.assertTrue(out.startswith(compact_prompt.COMPACT_PROMPT))
        self.assertTrue(out.endswith("Zentrix, Q-Lattice"))
        self.assertNotIn("literal dictation cleanup layer", out)

    def test_default_prompt_without_additions(self):
        self.assertEqual(compact_prompt.compact_system_prompt(FREEFLOW_DEFAULT), compact_prompt.COMPACT_PROMPT)

    def test_custom_prompt_is_not_recognised(self):
        self.assertIsNone(compact_prompt.compact_system_prompt("Clean this up like a pirate."))
        self.assertIsNone(compact_prompt.compact_system_prompt("You are a literal dictation cleanup layer. Short."))


class ServerPromptStyleTest(unittest.TestCase):
    def messages(self, system):
        return [{"role": "system", "content": system}, {"role": "user", "content": "RAW_TRANSCRIPTION"}]

    def test_swaps_only_in_compact_mode(self):
        with mock.patch.object(server, "LLM_PROMPT", "freeflow"):
            m = self.messages(FREEFLOW_DEFAULT + VOCAB_TAIL)
            server._use_prompt_style(m)
            self.assertEqual(m[0]["content"], FREEFLOW_DEFAULT + VOCAB_TAIL)
        with mock.patch.object(server, "LLM_PROMPT", "compact"):
            m = self.messages(FREEFLOW_DEFAULT + VOCAB_TAIL)
            server._use_prompt_style(m)
            self.assertTrue(m[0]["content"].startswith(compact_prompt.COMPACT_PROMPT))
            self.assertIn("Zentrix", m[0]["content"])
            custom = self.messages("Clean this up like a pirate.")
            server._use_prompt_style(custom)
            self.assertEqual(custom[0]["content"], "Clean this up like a pirate.")

    def test_adapter_path_needs_weights(self):
        self.assertIsNone(server.adapter_path(""))
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(server.adapter_path(d))
            (Path(d) / "adapters.safetensors").write_bytes(b"")
            self.assertEqual(server.adapter_path(d), d)


if __name__ == "__main__":
    unittest.main()
