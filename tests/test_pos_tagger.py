import pytest

spacy = pytest.importorskip("spacy")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not spacy.util.is_package("pl_core_news_sm"),
        reason="pl_core_news_sm spaCy model not installed",
    ),
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
