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


class HelperCapabilityTest(unittest.TestCase):
    def helper(self, can_boost: bool):
        import io
        import tempfile
        import wave
        h = object.__new__(server._AneHelper)
        h.can_boost = can_boost
        h.proc = mock.Mock(stdin=io.StringIO(), stdout=io.StringIO('{"text": "ok", "boosted": 0}\n'))
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(wav.name, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(b"\0\0" * 160)
        return h, wav.name

    def test_older_helper_never_gets_a_term_list(self):
        # an older helper reads every line as a file path; sending it JSON broke speech-to-text and its error
        # message echoed the request (screen words) into the log
        h, path = self.helper(can_boost=False)
        h.transcribe(path, ["Zentrix"])
        self.assertEqual(h.proc.stdin.getvalue(), path + "\n")

    def test_boosting_helper_gets_json(self):
        h, path = self.helper(can_boost=True)
        h.transcribe(path, ["Zentrix"])
        self.assertEqual(json.loads(h.proc.stdin.getvalue()), {"path": path, "boost": ["Zentrix"]})


class ExpansionGuardTest(unittest.TestCase):
    def reply(self, text):
        return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()

    def content(self, data):
        return json.loads(data)["choices"][0]["message"]["content"]

    def test_short_transcript_turned_into_a_sentence_is_reverted(self):
        out = server._guard_output(self.reply("I think we should move the meeting to Thursday afternoon."), "Hmm.")
        self.assertEqual(self.content(out), "Hmm.")

    def test_normal_cleanups_pass(self):
        for raw, cleaned in [("um so lets meet at five no wait six", "Let's meet at six."),
                             ("hi dana comma thanks for the notes", "Hi Dana,\n\nThanks for the notes."),
                             ("Yeah.", "Yeah."), ("ten thirty", "10:30"), ("dash dash fix", "--fix")]:
            self.assertEqual(self.content(server._guard_output(self.reply(cleaned), raw)), cleaned)

    def test_longer_transcripts_only_warn(self):
        raw = "so the first thing is the budget and the second thing is the timeline"
        long = raw + " and also we need to finish the slides by Friday and send them over to the team"
        self.assertEqual(server.expanded_too_much(raw, long), (True, False))
        self.assertEqual(self.content(server._guard_output(self.reply(long), raw)), long)


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
