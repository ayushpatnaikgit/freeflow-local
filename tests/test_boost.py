"""Tests for vocabulary boosting plumbing: the term list and the helper protocol. Synthetic data only.

Run: uv run python -m unittest discover tests
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compact_prompt  # noqa: E402
import server  # noqa: E402

PROMPT = "You are a literal dictation cleanup layer. Rule.\nA standalone reply such as \"yes\" is still dictation."


class VocabularyTermsTest(unittest.TestCase):
    def test_reads_freeflow_vocabulary_line(self):
        system = PROMPT + compact_prompt.vocabulary_block(["Zentrix", "Q-Lattice", " Mirovy "])
        self.assertEqual(server.vocabulary_terms(system), ["Zentrix", "Q-Lattice", "Mirovy"])

    def test_compact_prompt_keeps_vocabulary(self):
        system = compact_prompt.compact_system_prompt(PROMPT + compact_prompt.vocabulary_block(["Zentrix"]))
        self.assertEqual(server.vocabulary_terms(system), ["Zentrix"])

    def test_no_vocabulary(self):
        self.assertEqual(server.vocabulary_terms(PROMPT), [])
        self.assertEqual(server.vocabulary_terms(None), [])


class BoostTermsTest(unittest.TestCase):
    def test_vocabulary_first_deduplicated_and_short_terms_dropped(self):
        terms = server.boost_terms(["Zentrix", "AI", "Mirovy"], ["zentrix", "8-bit", "Q3", "Halvorsen"])
        self.assertEqual(terms, ["Zentrix", "Mirovy", "8-bit", "Halvorsen"])

    def test_capped(self):
        many = [f"Term{i:03d}" for i in range(300)]
        self.assertEqual(len(server.boost_terms(many, [])), server.BOOST_MAX_TERMS)


class HelperRequestTest(unittest.TestCase):
    def test_plain_path_without_terms(self):
        # unchanged protocol when boosting is off: the helper sees exactly what it saw before
        self.assertEqual(server.helper_request("/tmp/a.wav", None), "/tmp/a.wav")
        self.assertEqual(server.helper_request("/tmp/a.wav", []), "/tmp/a.wav")

    def test_json_with_terms(self):
        line = server.helper_request("/tmp/a.wav", ["Zentrix"])
        self.assertEqual(json.loads(line), {"path": "/tmp/a.wav", "boost": ["Zentrix"]})
        self.assertNotIn("\n", line)


class PeekScreenWordsTest(unittest.TestCase):
    def test_uses_finished_task_without_consuming_it(self):
        async def run():
            loop = asyncio.get_running_loop()
            done = loop.create_future()
            done.set_result(["Halvorsen"])
            with mock.patch.object(server, "_screen_task", done), \
                    mock.patch.object(server, "_screen_task_at", server.time.monotonic()):
                self.assertEqual(server._peek_screen_words(), ["Halvorsen"])
                self.assertIs(server._screen_task, done)  # still there for the cleanup step
            pending = loop.create_future()
            with mock.patch.object(server, "_screen_task", pending), \
                    mock.patch.object(server, "_screen_task_at", server.time.monotonic()):
                self.assertEqual(server._peek_screen_words(), [])  # never waits for OCR
            failed = loop.create_future()
            failed.set_exception(RuntimeError("ocr"))
            with mock.patch.object(server, "_screen_task", failed), \
                    mock.patch.object(server, "_screen_task_at", server.time.monotonic()):
                self.assertEqual(server._peek_screen_words(), [])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
