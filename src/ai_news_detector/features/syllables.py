import re

_WORD_RE = re.compile(r"\b\w+\b")
_POLISH_VOWEL_GROUPS_RE = re.compile(r"[aeiouyąęó]+", re.IGNORECASE)


def count_syllables_word(word: str) -> int:
    """Rough Polish syllable count: consecutive vowels count as one syllable nucleus."""
    letters = "".join(c for c in word if c.isalpha())
    if not letters:
        return 0
    groups = len(_POLISH_VOWEL_GROUPS_RE.findall(letters.lower()))
    return max(groups, 1)


def avg_syllables_per_sentence(text: str) -> float:
    """Mean total syllables per sentence; sentences split on '.' only (fragment between dots)."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0
    per_sentence: list[float] = []
    for s in sentences:
        words = _WORD_RE.findall(s)
        per_sentence.append(float(sum(count_syllables_word(w) for w in words)))
    return sum(per_sentence) / len(sentences)


def avg_word_length(text: str) -> float:
    """Mean character length of words (tokens matching ``\\w+``)."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)
