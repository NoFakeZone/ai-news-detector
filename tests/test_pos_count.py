import pytest
from unittest.mock import MagicMock

from ai_news_detector.features.pos import PosTag, PosTagCountExtractor


def _mock_tagger(tagged):
    m = MagicMock()
    m.tag.return_value = tagged
    return m


@pytest.mark.parametrize("tagged, target_tag, expected", [
    ([("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], PosTag.NOUN, 1.0),
    ([("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], PosTag.VERB, 1.0),
    ([("Kot", "NOUN"), ("pies", "NOUN"), ("biegnie", "VERB")], PosTag.NOUN, 2.0),
    ([("Kot", "NOUN"), ("biegnie", "VERB")], PosTag.ADJ, 0.0),
    ([], PosTag.NOUN, 0.0),
    ([("a", "ADJ"), ("b", "ADJ"), ("c", "ADJ")], PosTag.ADJ, 3.0),
])
def test_extract_counts_matching_tag(tagged, target_tag, expected):
    extractor = PosTagCountExtractor(tag=target_tag, tagger=_mock_tagger(tagged))
    assert extractor.extract("irrelevant") == expected


def test_extract_empty_text():
    spy = _mock_tagger([])
    extractor = PosTagCountExtractor(tagger=spy)
    assert extractor.extract("") == 0.0
    spy.tag.assert_not_called()


def test_extract_whitespace_only():
    spy = _mock_tagger([])
    extractor = PosTagCountExtractor(tagger=spy)
    assert extractor.extract("   \t\n  ") == 0.0
    spy.tag.assert_not_called()


def test_extract_returns_float():
    extractor = PosTagCountExtractor(
        tag=PosTag.NOUN,
        tagger=_mock_tagger([("Kot", "NOUN")]),
    )
    assert isinstance(extractor.extract("Kot"), float)


def test_tagger_called_once_with_text():
    spy = _mock_tagger([("Kot", "NOUN"), ("biegnie", "VERB")])
    extractor = PosTagCountExtractor(tag=PosTag.NOUN, tagger=spy)
    extractor.extract("Kot biegnie")
    spy.tag.assert_called_once_with("Kot biegnie")


@pytest.mark.parametrize("tag", list(PosTag))
def test_name_for_each_tag(tag):
    extractor = PosTagCountExtractor(tag=tag)
    assert extractor.name == f"pos_{tag.value.lower()}_count"


def test_punct_tag_produces_readable_name():
    extractor = PosTagCountExtractor(tag=PosTag.PUNCT)
    assert extractor.name == "pos_punct_count"
