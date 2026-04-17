import pytest

pytest.importorskip("spacy")

try:
    import spacy
    spacy.load("pl_core_news_sm")
    _has_model = True
except Exception:
    _has_model = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _has_model, reason="pl_core_news_sm spaCy model not installed"),
]


def test_spacy_polish_tagger_returns_ud_tags():
    from ai_news_detector.features.pos import SpacyPolishPosTagger

    tagger = SpacyPolishPosTagger()
    result = tagger.tag("Kot biegnie szybko.")
    tokens = {tok for tok, _ in result}
    tags = {tag for _, tag in result}
    assert "Kot" in tokens
    assert "NOUN" in tags
    assert "VERB" in tags
