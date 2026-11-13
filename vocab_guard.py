"""Undo vocabulary terms that the cleanup model put in place of words that don't sound like them.

A small fine-tuned model given a vocabulary list sometimes forces a term over unrelated words ("my M1 MacBook Air"
-> "my Q-Lattice"). The rule here is the one screen-word hints already follow: a vocabulary term may only replace
words that sound or look like it. After cleanup, every vocabulary term in the output that wasn't in the transcript is
traced back (word alignment) to the transcript words it replaced; if those don't sound like the term, they go back.
"""

import re
from difflib import SequenceMatcher

import screen_words

MIN_SIMILARITY = 0.5  # screen_words' sound+spelling score; real mishearings of terms score ~0.6-1.0
_CUE = re.compile(r"\b(no actually|actually|no wait|wait|sorry|i mean|scratch that|or rather|make that|no)\b[,.]?", re.I)


def _tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, normalised) for each word, keeping positions so the text can be patched in place."""
    return [(m.start(), m.end(), m.group().lower().strip(".,!?;:\"'()[]")) for m in re.finditer(r"\S+", text)]


def restore_unsupported_terms(raw: str, cleaned: str, terms: list[str]) -> tuple[str, int]:
    """Returns (text, number of spans restored). Only spans containing a vocabulary term not present in the raw
    transcript are touched; everything else the cleanup did stays."""
    if not terms or not raw or not cleaned:
        return cleaned, 0
    raw_lower = raw.lower()
    candidates = [t for t in terms if t.lower() not in raw_lower and re.search(re.escape(t), cleaned, re.I)]
    if not candidates:
        return cleaned, 0
    a, b = _tokens(raw), _tokens(cleaned)
    ops = SequenceMatcher(None, [w for *_, w in a], [w for *_, w in b], autojunk=False).get_opcodes()
    patches = []  # (start, end, replacement) in cleaned
    for tag, i1, i2, j1, j2 in ops:
        if tag not in ("replace", "insert") or j1 == j2:
            continue
        span = cleaned[b[j1][0]:b[j2 - 1][1]]
        term = next((t for t in candidates if re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", span, re.I)), None)
        if term is None:
            continue
        heard = " ".join(raw[s:e] for s, e, _ in a[i1:i2])
        # In a self-correction ("parakeet, no actually, or a keet"), the term replaces what comes after the last cue.
        cues = list(_CUE.finditer(heard))
        meant = heard[cues[-1].end():].strip(" ,.") if cues else heard
        if meant and screen_words._similarity(term, meant) >= MIN_SIMILARITY:
            continue  # a plausible mishearing of the term: keep the fix
        patches.append((b[j1][0], b[j2 - 1][1], heard))
    for start, end, replacement in sorted(patches, reverse=True):
        cleaned = cleaned[:start] + replacement + cleaned[end:]
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), len(patches)
