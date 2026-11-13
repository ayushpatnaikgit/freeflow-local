"""Learn vocabulary from the user's corrections of pasted dictation (opt-in: freeflowctl learn on).

edit-watch reports how the pasted text reads after the user edited it. This module compares the two word by word and
keeps a correction only if it looks like a mishearing of a name or term, in the spirit of OpenWhispr's auto-learn:

  - fewer than half of the words changed (otherwise it's a rewrite, not a correction);
  - the new text is short (1-3 words), has at least 3 letters or digits, and isn't a capitalisation-only change
    (unless the new spelling has inner capitals or digits, like GitHub or M3);
  - it contains an unusual word: not in the system dictionary, or a name/term by its shape (screen_words' rules);
  - old and new sound or look alike (sound-matcher, or spelling edit distance <= 65% of the length), so a change of
    mind ("Tuesday" -> "Wednesday") isn't learned.

Learned terms join boosting and the vocabulary hints. A misheard form becomes an exact alias (replaced right after
speech-to-text) once it has been corrected twice, or at once if none of its words is ordinary ("AuroKit").
Only terms and misheard forms are stored, in learned_vocabulary.json (mode 600); never the surrounding text.
"""

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import screen_words

MAX_RUN = 3  # words on either side of one correction


def _words(text: str) -> list[str]:
    return [w for w in (t.strip(".,!?;:\"'()[]{}") for t in text.split()) if w]


def corrections(pasted: str, edited: str) -> list[tuple[str, str]]:
    """(heard, corrected) pairs: runs of up to MAX_RUN words replaced by up to MAX_RUN words."""
    a, b = _words(pasted), _words(edited)
    if not a or not b:
        return []
    ops = SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False).get_opcodes()
    changed = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag != "equal")
    if changed > len(a) * 0.5:
        return []
    pairs = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "replace" and i2 - i1 <= MAX_RUN and j2 - j1 <= MAX_RUN:
            pairs.append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))
        elif tag == "equal" and i2 - i1 == j2 - j1:  # same words, different capitalisation (github -> GitHub)
            pairs += [(x, y) for x, y in zip(a[i1:i2], b[j1:j2]) if x != y]
    return pairs


def _edit_ratio(x: str, y: str) -> float:
    return 1 - SequenceMatcher(None, x.lower(), y.lower()).ratio()


def worth_learning(heard: str, corrected: str, known: set[str]) -> bool:
    letters = re.sub(r"[^A-Za-z0-9]", "", corrected)
    if len(letters) < 3 or len(corrected.split()) > MAX_RUN:
        return False
    if heard.lower() == corrected.lower():
        return bool(re.search(r"[a-z][A-Z]|\d", corrected))  # GitHub, M3: the spelling itself is the point
    if not any(screen_words._strength(w, known) for w in corrected.split()):
        return False  # only ordinary words: a content edit, not vocabulary
    return screen_words._similarity(corrected, heard) >= 0.5 or _edit_ratio(heard, corrected) <= 0.65


class LearnedVocabulary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = {"version": 1, "terms": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except (ValueError, OSError):
                pass

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def learn(self, pasted: str, edited: str, vocabulary: list[str]) -> int:
        """Record worthwhile corrections; returns how many were recorded. Terms already in FreeFlow's vocabulary
        still record their misheard forms (for aliases) but aren't listed as learned terms twice."""
        known = screen_words._known()
        found = [(h, c) for h, c in corrections(pasted, edited) if worth_learning(h, c, known)]
        if not found:
            return 0
        terms = self.data["terms"]
        for heard, term in found:
            key = next((k for k in terms if k.lower() == term.lower()), term)
            entry = terms.setdefault(key, {"count": 0, "heard": {}, "in_vocabulary": False})
            entry["count"] += 1
            entry["in_vocabulary"] = any(v.lower() == term.lower() for v in vocabulary)
            if heard.lower() != term.lower():
                entry["heard"][heard.lower()] = entry["heard"].get(heard.lower(), 0) + 1
        self._save()
        return len(found)

    def terms(self) -> list[str]:
        return [t for t, e in self.data["terms"].items() if not e.get("in_vocabulary")]

    def aliases(self) -> dict[str, str]:
        """Misheard form -> term, for forms corrected twice or that aren't ordinary words."""
        known = screen_words._known()
        out = {}
        for term, entry in self.data["terms"].items():
            for heard, n in entry["heard"].items():
                # every word unusual ("aurokit"), so the alias can't fire on ordinary speech ("graph ana" needs 2)
                unusual = all(screen_words._strength(w, known) or w.lower() not in known for w in heard.split())
                if n >= 2 or unusual:
                    out[heard] = term
        return out

    def forget(self, term: str) -> bool:
        key = next((k for k in self.data["terms"] if k.lower() == term.lower()), None)
        if key is None:
            return False
        del self.data["terms"][key]
        self._save()
        return True

    def clear(self) -> None:
        self.data["terms"] = {}
        self._save()


def apply_aliases(text: str, aliases: dict[str, str]) -> tuple[str, int]:
    """Replace whole-word misheard forms (case-insensitive), longest first. Returns (text, replacements)."""
    count = 0
    for heard in sorted(aliases, key=len, reverse=True):
        pattern = re.compile(r"(?<![\w-])" + r"\s+".join(map(re.escape, heard.split())) + r"(?![\w-])", re.I)
        text, n = pattern.subn(aliases[heard].replace("\\", r"\\"), text)
        count += n
    return text, count
