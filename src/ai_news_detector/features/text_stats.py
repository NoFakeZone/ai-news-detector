import re
import functools
from collections.abc import Callable

_WORD_RE = re.compile(r'\b\w+\b')
_SENTENCE_SEP_RE = re.compile(r'[.!?]+')

Lemmatizer = Callable[[str], list[str]]


@functools.cache
def _load_nlp():
    import spacy
    try:
        return spacy.load("pl_core_news_sm", disable=["ner", "parser"])
    except OSError as exc:
        raise RuntimeError(
            "Polish spaCy model not found. Install with: "
            "python -m spacy download pl_core_news_sm"
        ) from exc


def _default_lemmatize(text: str) -> list[str]:
    return [tok.lemma_ for tok in _load_nlp()(text) if tok.is_alpha]


def ttr(text: str) -> float:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def ttr_lemmatized(text: str, lemmatize: Lemmatizer = _default_lemmatize) -> float:
    if not text.strip():
        return 0.0
    lemmas = lemmatize(text.lower())
    if not lemmas:
        return 0.0
    return len(set(lemmas)) / len(lemmas)


def capital_ratio(text: str) -> float:
    if not text:
        return 0.0
    all_caps = len(re.findall(r'[A-Z]', text))
    sentence_starts = len(re.findall(r'(?:^|[.!?]+\s*)[A-Z]', text))
    internal_caps = max(0, all_caps - sentence_starts)
    return internal_caps / len(text)


def avg_sentence_len(text: str) -> float:
    sentences = [s.strip() for s in _SENTENCE_SEP_RE.split(text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(_WORD_RE.findall(s)) for s in sentences) / len(sentences)
