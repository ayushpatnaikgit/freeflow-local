"""Tests for learning vocabulary from corrections of pasted text. Invented text only.

Run: uv run python -m unittest discover tests
"""

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import screen_words  # noqa: E402
import server  # noqa: E402
import vocab_learner as V  # noqa: E402

def real_dictionary():
    # test_screen_words swaps in a tiny dictionary via the module global; these tests need the system one
    screen_words._dictionary = None
    return screen_words._known()


def learned(pasted, edited):
    known = real_dictionary()
    return [(h, c) for h, c in V.corrections(pasted, edited) if V.worth_learning(h, c, known)]


class RulesTest(unittest.TestCase):
    def test_learns_misheard_names_and_terms(self):
        self.assertEqual(learned("Ask Siobon to review it.", "Ask Siobhan to review it."), [("Siobon", "Siobhan")])
        self.assertEqual(learned("We moved the job to cubectal.", "We moved the job to kubectl."),
                         [("cubectal", "kubectl")])
        self.assertEqual(learned("We should use AuroKit for speech.", "We should use Zentrix for speech."), [])
        self.assertEqual(learned("We should use or who keat for speech.", "We should use Orukeet for speech."),
                         [("or who keat", "Orukeet")])
        self.assertEqual(learned("Push it to github tonight.", "Push it to GitHub tonight."), [("github", "GitHub")])

    def test_ignores_content_edits_and_rewrites(self):
        self.assertEqual(learned("Meet on Tuesday at noon.", "Meet on Wednesday at noon."), [])  # change of mind
        self.assertEqual(learned("I think their going.", "I think they are going."), [])  # ordinary words
        self.assertEqual(learned("the cat sat", "the dog sat"), [])
        self.assertEqual(learned("Send it to Siobon.", "Never mind, I will call Siobhan about the whole thing later."),
                         [])  # more than half changed: a rewrite
        self.assertEqual(learned("Meet at noon.", "Meet at noon."), [])
        self.assertEqual(learned("Hello Sam.", "hello Sam."), [])  # capitalisation of an ordinary word


class StoreTest(unittest.TestCase):
    def setUp(self):
        real_dictionary()
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "learned_vocabulary.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_learn_persist_and_forget(self):
        store = V.LearnedVocabulary(self.path)
        self.assertEqual(store.learn("Ask Siobon to review it.", "Ask Siobhan to review it.", []), 1)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        saved = json.loads(self.path.read_text())
        self.assertEqual(list(saved["terms"]), ["Siobhan"])
        self.assertNotIn("review", self.path.read_text())  # only the term and the misheard form are stored
        again = V.LearnedVocabulary(self.path)
        self.assertEqual(again.terms(), ["Siobhan"])
        self.assertTrue(again.forget("siobhan"))
        self.assertEqual(V.LearnedVocabulary(self.path).terms(), [])

    def test_terms_already_in_vocabulary_only_add_aliases(self):
        store = V.LearnedVocabulary(self.path)
        store.learn("We moved the job to cubectal.", "We moved the job to kubectl.", ["kubectl"])
        self.assertEqual(store.terms(), [])
        self.assertEqual(store.aliases(), {"cubectal": "kubectl"})

    def test_alias_needs_an_unusual_form_or_two_corrections(self):
        store = V.LearnedVocabulary(self.path)
        store.learn("Check the graph Ana dashboard.", "Check the Grafana dashboard.", [])
        self.assertEqual(store.aliases(), {})  # "graph ana" is made of ordinary words: once isn't enough
        store.learn("Open the graph Ana panel.", "Open the Grafana panel.", [])
        self.assertEqual(store.aliases(), {"graph ana": "Grafana"})

    def test_apply_aliases(self):
        text, n = V.apply_aliases("Ask SIOBON about cubectal, not cubectally.", {"siobon": "Siobhan", "cubectal": "kubectl"})
        self.assertEqual((text, n), ("Ask Siobhan about kubectl, not cubectally.", 2))
        self.assertEqual(V.apply_aliases("The graph  Ana view", {"graph ana": "Grafana"}), ("The Grafana view", 1))


class PastedTextTest(unittest.TestCase):
    def reply(self, text):
        return json.dumps({"choices": [{"message": {"content": text}}]}).encode()

    def test_matches_what_freeflow_pastes(self):
        self.assertEqual(server.pasted_text(self.reply('  "Hello there."  ')), "Hello there.")
        self.assertIsNone(server.pasted_text(self.reply("EMPTY")))
        self.assertIsNone(server.pasted_text(self.reply("")))
        self.assertIsNone(server.pasted_text(b"not json"))


if __name__ == "__main__":
    unittest.main()


class WatcherPlumbingTest(unittest.TestCase):
    def test_reported_edit_is_learned(self):
        # a stand-in for bin/edit-watch: answers each watch with the pasted text after a "user correction"
        import os
        import time
        from unittest import mock
        fake = Path(tempfile.mkdtemp()) / "edit-watch"
        fake.write_text("#!/usr/bin/python3\n"
                        "import json, sys\n"
                        "print(json.dumps({'ready': True, 'trusted': True}), flush=True)\n"
                        "for line in sys.stdin:\n"
                        "    msg = json.loads(line)\n"
                        "    if 'watch' in msg:\n"
                        "        print(json.dumps({'edited': msg['watch'].replace('Siobon', 'Siobhan')}), flush=True)\n")
        os.chmod(fake, 0o755)
        real_dictionary()
        store = V.LearnedVocabulary(Path(tempfile.mkdtemp()) / "learned.json")
        with mock.patch.object(server, "EDIT_WATCH", fake), mock.patch.object(server, "VOCAB_LEARN", True), \
                mock.patch.object(server, "_learned", store), mock.patch.object(server, "_watcher", None), \
                mock.patch.object(server, "_watcher_failed", False):
            watcher = server._edit_watcher()
            watcher.watch("Ask Siobon to review it.", [])
            watcher.watch("Nothing to fix here.", [])
            for _ in range(50):
                if store.terms():
                    break
                time.sleep(0.05)
            watcher.close()
        self.assertEqual(store.terms(), ["Siobhan"])
        self.assertEqual(store.aliases(), {"siobon": "Siobhan"})
