import functools
from collections.abc import Callable

from ai_news_detector.features.text_utils import count_words

Tagger = Callable[[str], list[tuple[str, str]]]

UD_TAGS: frozenset[str] = frozenset({
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE",
})


@functools.cache
def _load_nlp():
    import spacy
    try:
        return spacy.load("pl_core_news_sm", disable=["parser", "ner", "lemmatizer"])
    except OSError as exc:
        raise RuntimeError(
            "Polish spaCy model not found. Install with: "
            "python -m spacy download pl_core_news_sm"
        ) from exc

@functools.lru_cache(maxsize=2048)
def default_tagger(text: str) -> list[tuple[str, str]]:
    return [(tok.text, tok.pos_) for tok in _load_nlp()(text)]


def pos_count(
    text: str,
    tag: str = "NOUN",
    tagger: Tagger = default_tagger,
    validate: bool = False,
) -> float:
    if validate and tag not in UD_TAGS:
        raise ValueError(f"Unknown UD tag {tag!r}. Valid tags: {sorted(UD_TAGS)}")
    if not text.strip():
        return 0.0
    return float(sum(1 for _, t in tagger(text) if t == tag))


def pos_per_word(
    text: str,
    tag: str = "NOUN",
    tagger: Tagger = default_tagger,
    validate: bool = False,
) -> float:
    if validate and tag not in UD_TAGS:
        raise ValueError(f"Unknown UD tag {tag!r}. Valid tags: {sorted(UD_TAGS)}")
    words = count_words(text)
    if words == 0:
        return 0.0
    return pos_count(text, tag, tagger) / words

from collections import Counter

def all_pos_per_word(
    text: str,
    tagger: Tagger = default_tagger
) -> dict[str, float]:
    """
    Calculates the ratio for ALL Universal Dependency tags in a single pass.
    Returns a dictionary mapping 'TAG' -> ratio.
    """
    words = count_words(text)
    
    # Initialize all tags to 0.0 to ensure consistent output format
    ratios = {tag: 0.0 for tag in UD_TAGS}
    
    if words == 0 or not text.strip():
        return ratios

    # Run tagger ONCE
    tagged_words = tagger(text)
    
    # Count all tags simultaneously using Python's highly optimized Counter
    tag_counts = Counter(tag for _, tag in tagged_words)
    
    # Calculate ratios
    for tag, count in tag_counts.items():
        if tag in ratios: # Ensure we only track UD_TAGS
            ratios[tag] = count / words
            
    return ratios