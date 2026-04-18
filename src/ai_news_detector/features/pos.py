import functools
from collections.abc import Callable

from ai_news_detector.features.text_utils import count_words

Tagger = Callable[[str], list[tuple[str, str]]]


@functools.cache
def _load_nlp():
    import spacy
    try:
        return spacy.load("pl_core_news_sm")
    except OSError as exc:
        raise RuntimeError(
            "Polish spaCy model not found. Install with: "
            "python -m spacy download pl_core_news_sm"
        ) from exc


def default_tagger(text: str) -> list[tuple[str, str]]:
    return [(tok.text, tok.pos_) for tok in _load_nlp()(text)]


def pos_count(text: str, tag: str = "NOUN", tagger: Tagger = default_tagger) -> float:
    if not text.strip():
        return 0.0
    return float(sum(1 for _, t in tagger(text) if t == tag))


def pos_per_word(text: str, tag: str = "NOUN", tagger: Tagger = default_tagger) -> float:
    words = count_words(text)
    if words == 0:
        return 0.0
    return pos_count(text, tag, tagger) / words
