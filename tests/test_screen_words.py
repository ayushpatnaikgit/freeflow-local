"""Tests for screen_words.py. Run: uv run python -m unittest discover tests. All text is invented."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import screen_words  # noqa: E402


class ScreenWordsTest(unittest.TestCase):
    def setUp(self):
        screen_words._dictionary = {"the", "train", "adapter", "on", "with", "then", "serve", "it", "from", "ask",
                                    "to", "check", "dashboard", "after", "rollout", "restart", "file", "edit",
                                    "view", "help", "laura", "gemma", "model", "runs", "neural", "engine", "was",
                                    "worse", "at", "cleanup", "store", "embed", "in", "office", "is", "and", "review", "hand",
                                    "install", "readme", "explorer", "item"}

    def test_picks_rare_terms_and_skips_dictionary_words(self):
        lines = ["File Edit View Help",
                 "Train a LoRA adapter on Gemma with mlx_lm.lora, then serve it from MLX.",
                 "The Orukeet model runs on the Neural Engine; Qwen3 was worse at cleanup.",
                 "Ask Siddhant to check the Grafana dashboard after kubectl rollout restart.",
                 "We store embeddings in Postgres with pgvector; the Tiruchirappalli office is on Zulip."]
        words = screen_words.screen_words(lines * 2)  # the helper reads each region at least twice
        for expected in ["LoRA", "MLX", "Orukeet", "Qwen3", "Siddhant", "Grafana", "kubectl", "Postgres", "pgvector",
                         "Tiruchirappalli", "Zulip"]:
            self.assertIn(expected, words)
        for ordinary in ["File", "Edit", "Gemma", "Train", "Neural", "Engine", "dashboard"]:
            self.assertNotIn(ordinary, words)

    def test_ranks_by_frequency_and_keeps_most_common_spelling(self):
        words = screen_words.screen_words(["LoRA LoRA Lora-ish", "LoRA rank", "Zulip Zulip"])
        self.assertEqual(words, ["LoRA", "Zulip"])
        self.assertEqual(screen_words.screen_words(["FreeFlow freeflow FreeFlow"]), ["FreeFlow"])

    def test_skips_noise(self):
        self.assertEqual(screen_words.screen_words(["OK  A  42  3.5  https www com  .md", "PM AM ID"]), [])
        # file names, code identifiers and inflections of dictionary words
        self.assertEqual(screen_words.screen_words(
            ["README.md install.sh mlx_lm.lora _context_metadata", "embeddings handing reviewed"]), [])

    def test_strong_terms_rank_before_unknown_lowercase(self):
        words = screen_words.screen_words(["zzyzx zzyzx zzyzx zzyzx", "LoRA LoRA"])
        self.assertEqual(words, ["LoRA", "zzyzx"])

    def test_possessive_and_limit(self):
        self.assertEqual(screen_words.screen_words(["Siddhant's review", "Siddhant"]), ["Siddhant"])
        many = [f"Zork{i}" for i in range(100)] * 2
        self.assertEqual(len(screen_words.screen_words(many, limit=5)), 5)

    def test_single_reads_are_dropped_as_likely_misreads(self):
        self.assertEqual(screen_words.screen_words(["Tiruchirappali office", "Tiruchirappall office", "Zulip Zulip"]),
                         ["Zulip"])
        self.assertEqual(screen_words.screen_words(["Zulip"], min_reads=1), ["Zulip"])

    def test_file_names_with_lost_dots_are_dropped(self):
        self.assertEqual(screen_words.screen_words(["READMEmd Installsh", "READMEmd Installsh", "Bangladesh Bangladesh"]),
                         ["Bangladesh"])

    def test_relevant_keeps_only_words_heard_in_the_transcript(self):
        screen = ["LoRA", "MLX", "Orukeet", "Qwen3", "Siddhant", "Grafana", "Zulip", "exploreritem", "itern"]
        self.assertEqual(screen_words.relevant(screen, "I think the Laura fine tune might be good."), ["LoRA"])
        self.assertEqual(sorted(screen_words.relevant(screen, "Ask Sid hunt to check the graph Ana dashboard.")),
                         ["Grafana", "Siddhant"])
        self.assertEqual(screen_words.relevant(screen, "We should use or a keet, not quen three."), ["Orukeet", "Qwen3"])
        self.assertEqual(screen_words.relevant(screen, "So I was thinking we could move the standup to ten thirty."), [])
        self.assertEqual(screen_words.relevant(screen, "Mark the item as done and move on to the next one."), [])
        self.assertEqual(screen_words.relevant([], "anything"), [])

    def test_relevant_skips_words_already_spelled_right_and_other_names(self):
        self.assertEqual(screen_words.relevant(["Priya", "LoRA"], "Send it to Priya, sorry, to Pooja."), [])
        self.assertEqual(screen_words.relevant(["Siddhant"], "Ping Siddharth about the upgrade."), [])
        self.assertEqual(screen_words.relevant(["Nadia"], "Nadya, can you review this?"), ["Nadia"])
        self.assertEqual(screen_words.relevant(["LoRA"], "Train a Laura adapter."), ["LoRA"])

    def test_number_hyphen_terms_are_read_whole(self):
        # "8-bit" used to be read as "bit" (a dictionary word) and dropped, so "eighth bit" was never corrected.
        words = screen_words.screen_words(["Neural Engine 8-bit and 6-bit builds"] * 2)
        self.assertIn("8-bit", words)
        self.assertIn("6-bit", words)
        self.assertEqual(screen_words.relevant(words, "Why are we even doing the eighth bit?"), ["8-bit"])

    def test_opening_vowel_and_wh_are_loose(self):
        # recognisers swap opening vowels and spell "u" as "who": both still match the listed term
        self.assertEqual(screen_words.relevant(["Orukeet"], "We should use Aurokeet for speech."), ["Orukeet"])
        self.assertEqual(screen_words.relevant(["Orukeet"], "We should use or who keat for speech."), ["Orukeet"])
        self.assertEqual(screen_words.relevant(["Orukeet"], "Who is going to the meeting?"), [])

    def test_prompt_section(self):
        self.assertEqual(screen_words.prompt_section([]), "")
        section = screen_words.prompt_section(["LoRA", "MLX"])
        self.assertTrue(section.startswith("SCREEN_WORDS"))
        self.assertIn("LoRA, MLX", section)
        self.assertIn("never add a screen word that was not spoken", section)


if __name__ == "__main__":
    unittest.main()
