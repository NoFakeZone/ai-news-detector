"""Polish lemmatizer. spaCy is imported lazily so importing this module stays cheap and tests using mock lemmatizers do not require spaCy."""
from dataclasses import dataclass
from typing import Protocol

import functools


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


class Lemmatizer(Protocol):
    def lemmatize(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class SpacyPolishLemmatizer:
    def lemmatize(self, text: str) -> list[str]:
        return [token.lemma_ for token in _load_nlp()(text) if token.is_alpha]
