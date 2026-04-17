""" Polish POS tagger.

spaCy is imported lazily
so importing this module stays cheap
and tests using MagicMock taggers
do not require spaCy.
"""
import functools
from dataclasses import dataclass
from typing import Protocol


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


class PosTagger(Protocol):
    def tag(self, text: str) -> list[tuple[str, str]]: ...


@dataclass(frozen=True)
class SpacyPolishPosTagger:
    def tag(self, text: str) -> list[tuple[str, str]]:
        return [(tok.text, tok.pos_) for tok in _load_nlp()(text)]
