"""Pick the unusual words on screen (LoRA, MLX, Qwen3, Siddhant) from OCR text of FreeFlow's context screenshot.

Speech recognition turns rare words into common ones ("LoRA" -> "Laura", "Orukeet" -> "or a keet"). If the word is
on screen, cleanup can put the right spelling back. Ordinary English is left out: the dictionary already spells it,
and listing it would only invite the model to swap words. Nothing here is logged or stored.
"""

import difflib
import re
from collections import Counter
from pathlib import Path

DICTIONARY = Path("/usr/share/dict/words")
MAX_WORDS = 40
# The OCR helper reads every part of the screen several times (as drawn, inverted, overlapping tiles). A word read
# the same way at least twice is real; misreads of small text vary between reads. On synthetic windows this kept every
# correctly read planted word and cut the other words by more than half.
MIN_READS = 2
# Only screen words that sound like something in the transcript are sent (see relevant()). Unrelated screen words
# were not harmless: on a 77-case benchmark, a list of them made cleanup leave in fillers it otherwise removed.
MATCH_THRESHOLD = 0.75
MAX_SENT = 12

# Words with letters, digits and inner - + ' (Qwen3, gpt-4o, C++, mlx-lm, 8-bit); edges trimmed below. Tokens with a dot
# inside (install.sh, README.md, mlx_lm.lora) are file names or code, mostly sidebar noise, so they are dropped.
_TOKEN = re.compile(r"(?:\d+-)?[A-Za-z0-9_.]*[A-Za-z][A-Za-z0-9]*(?:[\-+'][A-Za-z0-9]+)*\+*[A-Za-z0-9_.]*")
# UI chrome and technical noise that is unusual in form but useless as dictation vocabulary.
_SKIP = {"ok", "id", "ui", "url", "urls", "http", "https", "www", "com", "md", "py", "sh", "js", "ts", "json", "html",
         "png", "jpg", "pdf", "txt", "app", "ctrl", "cmd", "esc", "tab", "pm", "am"}

# OCR often loses the dot in file names: "README.md" -> "READMEmd", "install.sh" -> "Installsh".
_GLUED_EXTENSION = re.compile(r"^(?P<stem>[A-Za-z]{3,}?)(?:md|sh|py|js|ts|txt|json|html|swift|yml|yaml|toml)$")

_dictionary: set[str] | None = None


def _known() -> set[str]:
    global _dictionary
    if _dictionary is None:
        try:
            _dictionary = {w.strip().lower() for w in DICTIONARY.read_text(errors="ignore").splitlines()}
        except OSError:
            _dictionary = set()
    return _dictionary


_SUFFIXES = ("ings", "ing", "ers", "er", "ed", "es", "s", "ly")


def _in_dictionary(word: str, known: set[str]) -> bool:
    """Known word, or a known word plus a common ending (embeddings, handing, rollouts)."""
    w = word.lower()
    if w in known:
        return True
    for suffix in _SUFFIXES:
        stem = w[: -len(suffix)]
        if w.endswith(suffix) and len(stem) >= 3 and (stem in known or stem + "e" in known
                                                       or (len(stem) > 3 and stem[-1] == stem[-2] and stem[:-1] in known)):
            return True
    return False


def _strength(word: str, known: set[str]) -> int:
    """2 = clearly a name or term (LoRA, MLX, Qwen3, Siddhant), 1 = unknown lowercase word (pgvector), 0 = skip."""
    letters = re.sub(r"[^A-Za-z]", "", word)
    if len(letters) < 2 or word.lower() in _SKIP or "_" in word or "." in word:
        return 0
    glued = _GLUED_EXTENSION.match(word)
    if glued and (glued["stem"].isupper() or _in_dictionary(glued["stem"], known)):
        return 0
    if re.search(r"[a-z][A-Z]", word):                       # LoRA, FreeFlow, iPhone
        return 2
    if re.search(r"\d", word):                               # Qwen3, gpt-4o, M4 (not bare numbers)
        return 2
    if word.isupper():                                       # MLX, LMAA (not single letters)
        return 2 if len(word) <= 8 else 0
    if len(letters) < 4 or _in_dictionary(word, known):
        return 0
    if word[0].isupper():                                    # Siddhant, Orukeet, Zulip
        return 2
    return 1 if len(letters) >= 5 else 0                     # kubectl, pgvector (and some OCR noise)


def screen_words(lines: list[str], limit: int = MAX_WORDS, min_reads: int = MIN_READS) -> list[str]:
    """Unusual words read at least `min_reads` times, strongest and most frequent first, one spelling each."""
    known = _known()
    counts: Counter[str] = Counter()
    strength: dict[str, int] = {}
    spelling: dict[str, Counter[str]] = {}
    for line in lines:
        for raw in _TOKEN.findall(line):
            word = raw.strip(".-'")
            if word.endswith("'s"):
                word = word[:-2]
            level = _strength(word, known) if word else 0
            if not level:
                continue
            key = word.lower()
            counts[key] += 1
            strength[key] = max(strength.get(key, 0), level)
            spelling.setdefault(key, Counter())[word] += 1
    ranked = sorted((k for k in counts if counts[k] >= min_reads), key=lambda k: (-strength[k], -counts[k], k))
    return [spelling[k].most_common(1)[0][0] for k in ranked[:limit]]


_NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10",
            # ordinals: "8-bit" is often heard as "eighth bit"
            "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
            "eighth": "8", "ninth": "9", "tenth": "10"}
_SOUNDS = (("wh", "h"), ("ph", "f"), ("ck", "k"), ("qu", "k"), ("q", "k"), ("c", "k"), ("x", "ks"), ("z", "s"), ("w", "v"),
           ("dh", "d"), ("th", "t"), ("gh", "g"))


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _sound(text: str) -> str:
    """Rough sound-alike key: similar consonants merged, vowels after the first letter dropped (LoRA = Laura)."""
    s = _letters(text)
    for a, b in _SOUNDS:
        s = s.replace(a, b)
    if not s:
        return s
    # any opening vowel counts as the same sound: recognisers often swap it ("AuroKit" for Orukeet)
    first = "a" if s[0] in "aeiou" else s[0]
    return re.sub(r"(.)\1+", r"\1", first + re.sub(r"[aeiouy]", "", s[1:]))


def _similarity(word: str, spoken: str) -> float:
    """Mean of sound-alike and spelling similarity; either alone matched too much ("or a" ~ LoRA by spelling,
    "prod" ~ Priya by sound)."""
    by_letters = difflib.SequenceMatcher(None, _letters(word), _letters(spoken)).ratio()
    a, b = _sound(word), _sound(spoken)
    by_sound = difflib.SequenceMatcher(None, a, b).ratio() if len(a) >= 2 else 0.0
    return (by_letters + by_sound) / 2


def relevant(words: list[str], transcript: str, threshold: float = MATCH_THRESHOLD, limit: int = MAX_SENT) -> list[str]:
    """The screen words that sound like a 1-3 word stretch of the transcript ("Laura" -> LoRA, "or a keet" -> Orukeet).

    Tuned on invented pairs: true pairs scored 0.75-1.0 except names spelled far from how they sound ("voy check" ->
    Wojciech, "wakeen" -> Joaquin), which FreeFlow's context summary can still supply.
    """
    tokens = [_NUMBERS.get(t.lower(), t) for t in re.findall(r"[A-Za-z0-9']+", transcript)]
    spans = [" ".join(tokens[i:i + n]) for n in (1, 2, 3) for i in range(len(tokens) - n + 1)]
    spoken = {_letters(t) for t in tokens}
    spoken_digits = {d for t in tokens for d in re.findall(r"\d+", t)}
    known = _known()
    scored = []
    for word in words:
        w = _letters(word)
        if w in spoken:  # already spelled this way in the transcript: nothing to fix
            continue
        # A number in a screen word must have been said: "eighth bit" can be 8-bit, never 6-bit.
        if any(d not in spoken_digits for d in re.findall(r"\d+", word)):
            continue
        best, span = max(((_similarity(word, span), span) for span in spans
                          if abs(len(_letters(span)) - len(w)) <= max(3, len(w) // 2)), default=(0.0, ""))
        if best < threshold:
            continue
        # The recogniser already wrote a different-sounding name here ("Siddharth" with "Siddhant" on screen):
        # trust it rather than swap one real name for another. Same-sounding spellings (Nadya/Nadia) still pass.
        if (" " not in span and span[:1].isupper() and not _in_dictionary(span, known)
                and _sound(span) != _sound(word)):
            continue
        scored.append((best, word))
    scored.sort(key=lambda x: -x[0])
    return [word for _, word in scored[:limit]]


def prompt_section(words: list[str]) -> str:
    """Text added to the cleanup request, next to FreeFlow's CONTEXT line."""
    if not words:
        return ""
    return ("SCREEN_WORDS (spellings visible on the user's screen, as read by OCR; may contain misreads): "
            + ", ".join(words) + "\n"
            "Use a SCREEN_WORDS spelling only to replace a word or phrase in RAW_TRANSCRIPTION that is clearly a "
            "mis-hearing of it, for example a name or technical term heard as ordinary words. Replace only what was "
            "misheard: never add a screen word that was not spoken (such as a surname next to a spoken first name), "
            "and keep ordinary words that make sense as they are.\n")
