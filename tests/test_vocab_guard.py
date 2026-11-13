"""Tests for undoing vocabulary terms forced over unrelated words. Invented text only.

Run: uv run python -m unittest discover tests
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import screen_words  # noqa: E402
import server  # noqa: E402
import vocab_guard as G  # noqa: E402


class VocabGuardTest(unittest.TestCase):
    def setUp(self):
        screen_words._dictionary = None  # other tests swap in a tiny dictionary

    def test_unrelated_words_come_back(self):
        # a 0.6B model with a vocabulary list put a term over words that sound nothing like it
        cases = [("Will this work well on my M one MacBook Air?", "Will this work well on my MOU?", ["MOU"],
                  "Will this work well on my M one MacBook Air?"),
                 ("Send the MOU to legal today.", "Send the MOU to Q-Lattice today.", ["MOU", "Q-Lattice"],
                  "Send the MOU to legal today."),
                 ("I bought an M one Mac mini last year.", "I bought an MOU last year.", ["MOU"],
                  "I bought an M one Mac mini last year.")]
        for raw, cleaned, terms, expected in cases:
            self.assertEqual(G.restore_unsupported_terms(raw, cleaned, terms), (expected, 1))

    def test_real_mishearings_and_corrections_are_kept(self):
        keep = [("We signed the M O U yesterday.", "We signed the MOU yesterday.", ["MOU"]),
                ("Check the graph Ana dashboard.", "Check the Grafana dashboard.", ["Grafana"]),
                ("We should use parakeet, no actually, or a keet for the speech model.",
                 "We should use Orukeet for the speech model.", ["Orukeet"]),
                ("Send the MOU to legal.", "Send the MOU to legal.", ["MOU"]),  # already said
                ("hi dana comma thanks", "Hi Dana,\n\nThanks", ["MOU"])]  # no term involved
        for raw, cleaned, terms in keep:
            self.assertEqual(G.restore_unsupported_terms(raw, cleaned, terms), (cleaned, 0))

    def test_server_applies_it_to_the_reply(self):
        reply = json.dumps({"choices": [{"message": {"content": "Will this work on my MOU?"}}]}).encode()
        out = server._guard_vocabulary(reply, "Will this work on my M one MacBook Air?", ["MOU"])
        self.assertEqual(json.loads(out)["choices"][0]["message"]["content"], "Will this work on my M one MacBook Air?")


if __name__ == "__main__":
    unittest.main()
