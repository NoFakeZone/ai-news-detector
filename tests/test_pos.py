import pytest
from ai_news_detector.features.pos import pos_count, pos_per_word, UD_TAGS

try:
    import spacy
    spacy.load("pl_core_news_sm")
    _has_model = True
except Exception:
    _has_model = False

_requires_model = pytest.mark.skipif(not _has_model, reason="pl_core_news_sm not installed")


def _tagger(pairs):
    return lambda text: pairs


@pytest.mark.parametrize("tagged, tag, expected", [
    ([("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], "NOUN", 1.0),
    ([("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], "VERB", 1.0),
    ([("Kot", "NOUN"), ("pies", "NOUN"), ("biegnie", "VERB")], "NOUN", 2.0),
    ([("Kot", "NOUN"), ("biegnie", "VERB")], "ADJ", 0.0),
    ([], "NOUN", 0.0),
])
def test_count(tagged, tag, expected):
    assert pos_count("text", tag=tag, tagger=_tagger(tagged)) == expected


def test_count_empty_text():
    tagger = _tagger([("Kot", "NOUN")])
    assert pos_count("", tagger=tagger) == 0.0


def test_count_whitespace_only():
    tagger = _tagger([("Kot", "NOUN")])
    assert pos_count("   ", tagger=tagger) == 0.0


def test_count_returns_float():
    assert isinstance(pos_count("text", tagger=_tagger([("x", "NOUN")])), float)


def test_count_validate_unknown_tag_raises():
    with pytest.raises(ValueError):
        pos_count("text", tag="BOGUS", validate=True)


def test_count_validate_known_tag_ok():
    pos_count("text", tag="NOUN", tagger=_tagger([]), validate=True)


def test_ud_tags_contains_common_tags():
    assert {"NOUN", "VERB", "ADJ", "ADV"}.issubset(UD_TAGS)


@pytest.mark.parametrize("text, tagged, expected", [
    ("Kot biegnie szybko", [("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], pytest.approx(1 / 3)),
    ("Kot biegnie", [("Kot", "NOUN"), ("biegnie", "VERB")], pytest.approx(0.5)),
    ("", [], 0.0),
    ("   ", [], 0.0),
])
def test_per_word(text, tagged, expected):
    assert pos_per_word(text, tag="NOUN", tagger=_tagger(tagged)) == expected


def test_per_word_returns_float():
    assert isinstance(pos_per_word("text", tagger=_tagger([("x", "NOUN")])), float)


@pytest.mark.slow
@_requires_model
def test_default_tagger_returns_ud_tags():
    from ai_news_detector.features.pos import default_tagger
    result = default_tagger("Kot biegnie szybko.")
    tags = {tag for _, tag in result}
    assert "NOUN" in tags
    assert "VERB" in tags
