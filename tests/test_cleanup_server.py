"""Tests for copy-ahead decoding in cleanup_server.py, with a fake deterministic model (no MLX, no real model).

Run: uv run python -m unittest discover tests
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cleanup_server as C  # noqa: E402

EOS = 0


class FakeModel:
    """Next token depends on the last two tokens; a KV 'cache' is just the list of tokens fed so far."""

    def __init__(self, table):
        self.table, self.cache = table, []

    def next_after(self, seq):
        return self.table[(seq[-2] if len(seq) > 1 else -1, seq[-1])]

    def predict(self, tokens):
        out = []
        for t in tokens:
            self.cache.append(t)
            out.append(self.next_after(self.cache))
        return out

    def trim(self, n):
        del self.cache[len(self.cache) - n:]


def greedy(model, prompt, max_tokens):
    seq, out = list(prompt), []
    while True:
        t = model.next_after(seq)
        if t == EOS:
            return out, False
        if len(out) >= max_tokens:
            return out, True
        out.append(t)
        seq.append(t)


def make_table(rng, vocab):
    keys = [(a, b) for a in range(-1, vocab) for b in range(vocab)]
    return {k: (EOS if rng.random() < 0.04 else rng.randrange(1, vocab)) for k in keys}


class CopyAheadTest(unittest.TestCase):
    def test_lookup_draft_copies_what_followed_the_last_match(self):
        self.assertEqual(C.lookup_draft([5, 6, 7, 8, 9, 5, 6], 3), [7, 8, 9])
        self.assertEqual(C.lookup_draft([1, 2, 3], 4), [])  # nothing repeats
        self.assertEqual(C.lookup_draft([4, 1, 2, 4], 2), [1, 2])  # falls back to a 1-token match

    def test_same_output_as_greedy_for_every_draft_length(self):
        rng = random.Random(7)
        for trial in range(300):
            vocab = rng.choice([4, 6, 12])
            table = make_table(rng, vocab)
            prompt = [rng.randrange(1, vocab) for _ in range(rng.randrange(2, 30))]
            max_tokens = rng.choice([1, 5, 40])
            want = greedy(FakeModel(table), prompt, max_tokens)
            for k in (0, 1, 2, 4, 8):
                m = FakeModel(table)
                first = m.predict(prompt)[-1]
                got = C.lookahead_decode(m.predict, m.trim, first, prompt, {EOS}, max_tokens, k)
                self.assertEqual(got, want, f"trial {trial}, k={k}")
                # the cache holds exactly the prompt and the output, so the next request can reuse it
                self.assertEqual(m.cache, prompt + got[0], f"trial {trial}, k={k}")

    def test_drafts_are_accepted_when_output_repeats_the_prompt(self):
        # a model that copies the prompt: next token is the one after the previous token in the prompt
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
        follow = {prompt[i]: prompt[i + 1] for i in range(len(prompt) - 1)}
        follow[9] = EOS
        table = {(a, b): follow.get(b, EOS) for a in range(-1, 10) for b in range(10)}
        calls = []
        m = FakeModel(table)
        first = m.predict(prompt)[-1]

        def predict(tokens):
            calls.append(len(tokens))
            return m.predict(tokens)

        out, _ = C.lookahead_decode(predict, m.trim, first, prompt, {EOS}, 100, 4)
        self.assertEqual(out, [2, 3, 4, 5, 6, 7, 8, 9])
        self.assertLess(len(calls), len(out))  # fewer forward passes than tokens


if __name__ == "__main__":
    unittest.main()
