"""Tests for how server.py finds the screenshot and adds screen words to a cleanup request. Synthetic data only.

Run: uv run python -m unittest discover tests
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402

CLEANUP_USER = ("Instructions: Clean up RAW_TRANSCRIPTION and return only the cleaned transcript text.\n\n"
                "CONTEXT: \"The user is in a code editor.\"\n\n\nRAW_TRANSCRIPTION:\n<<<RAW_TRANSCRIPTION\n"
                "Train a Laura adapter.\nRAW_TRANSCRIPTION")


def done(words):
    future = asyncio.get_running_loop().create_future()
    future.set_result(words)
    return future


class ScreenWordsServerTest(unittest.TestCase):
    def test_finds_data_url_screenshot_only(self):
        url = "data:image/jpeg;base64,AAAA"
        context = [{"role": "system", "content": "x"},
                   {"role": "user", "content": [{"type": "text", "text": "App: Code"},
                                                {"type": "image_url", "image_url": {"url": url}}]}]
        self.assertEqual(server._screenshot_url(context), url)
        remote = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]
        self.assertIsNone(server._screenshot_url(remote))
        self.assertIsNone(server._screenshot_url([{"role": "user", "content": "text only"}]))
        self.assertIsNone(server._screenshot_url([]))

    def test_adds_words_to_user_message_not_system_prompt(self):
        async def go():
            messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": CLEANUP_USER}]
            await server._with_screen_words(messages, done(["LoRA", "MLX", "Grafana"]), "Train a Laura adapter.")
            return messages
        messages = asyncio.run(go())
        self.assertEqual(messages[0]["content"], "SYSTEM")
        user = messages[1]["content"]
        self.assertIn("SCREEN_WORDS", user)
        self.assertIn("SCREEN_WORDS (", user)
        self.assertIn("): LoRA\n", user)  # only the word that sounds like part of the transcript
        self.assertLess(user.index("SCREEN_WORDS"), user.index("RAW_TRANSCRIPTION:\n<<<"))
        self.assertTrue(user.endswith("Train a Laura adapter.\nRAW_TRANSCRIPTION"))

    def test_no_words_or_no_task_leaves_request_unchanged(self):
        async def go(task_words):
            messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": CLEANUP_USER}]
            await server._with_screen_words(messages, None if task_words is None else done(task_words),
                                            "Train a Laura adapter.")
            return messages[1]["content"]
        self.assertEqual(asyncio.run(go(None)), CLEANUP_USER)
        self.assertEqual(asyncio.run(go([])), CLEANUP_USER)
        self.assertEqual(asyncio.run(go(["Grafana", "Siddhant"])), CLEANUP_USER)  # nothing that sounds like it

    def test_slow_ocr_is_skipped(self):
        async def go():
            slow = asyncio.get_running_loop().create_future()  # never finishes
            messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": CLEANUP_USER}]
            saved, server.SCREEN_WORDS_WAIT = server.SCREEN_WORDS_WAIT, 0.05
            try:
                await server._with_screen_words(messages, slow, "Train a Laura adapter.")
            finally:
                server.SCREEN_WORDS_WAIT = saved
            return messages[1]["content"]
        self.assertEqual(asyncio.run(go()), CLEANUP_USER)

    def test_task_is_used_once_and_expires(self):
        async def go():
            server._screen_task, server._screen_task_at = done(["LoRA"]), server.time.monotonic()
            first, second = server._take_screen_task(), server._take_screen_task()
            server._screen_task = done(["LoRA"])
            server._screen_task_at = server.time.monotonic() - server.SCREEN_WORDS_MAX_AGE - 1
            stale = server._take_screen_task()
            return first, second, stale
        first, second, stale = asyncio.run(go())
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertIsNone(stale)


if __name__ == "__main__":
    unittest.main()


class SecurityTest(unittest.TestCase):
    def test_requests_from_web_pages_are_refused(self):
        page = {"origin": "https://example.com", "sec-fetch-site": "cross-site"}
        self.assertTrue(server._from_web_page(page))
        self.assertTrue(server._from_web_page({"origin": "http://127.0.0.1:3000"}))
        self.assertTrue(server._from_web_page({"sec-fetch-site": "same-site"}))
        self.assertFalse(server._from_web_page({}))  # FreeFlow, curl, freeflowctl
        self.assertFalse(server._from_web_page({"sec-fetch-site": "none"}))  # typing the URL in a browser

    def test_middleware_answers_403_to_a_web_page(self):
        async def call(headers):
            sent = []
            scope = {"type": "http", "method": "GET", "path": "/health", "raw_path": b"/health", "query_string": b"",
                     "headers": [(k.encode(), v.encode()) for k, v in headers.items()], "http_version": "1.1",
                     "scheme": "http", "server": ("127.0.0.1", 8081), "client": ("127.0.0.1", 5000), "root_path": ""}

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                sent.append(message)
            await server.app(scope, receive, send)
            return next(m["status"] for m in sent if m["type"] == "http.response.start")
        self.assertEqual(asyncio.run(call({"origin": "https://example.com"})), 403)
        self.assertEqual(asyncio.run(call({})), 200)

    def test_pins_file_parsing(self):
        pins = server.read_pins("# comment\nowner/model abc123  # trailing\n\nbad line here x\nother/m def\n")
        self.assertEqual(pins, {"owner/model": "abc123", "other/m": "def"})
        shipped = server.read_pins((server.HERE / "pinned-models.txt").read_text())
        self.assertIn("mlx-community/gemma-4-e4b-it-4bit", shipped)
        self.assertTrue(all(len(v) == 40 for v in shipped.values()))
