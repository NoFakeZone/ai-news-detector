import pytest
from unittest.mock import MagicMock

from ai_news_detector.features.pos import (
    PosTag,
    PosTagCountExtractor,
    PosTagPerWordExtractor,
)


def _mock_counter(return_value):
    m = MagicMock()
    m.extract.return_value = return_value
    return m


@pytest.mark.parametrize("text, counter_return, expected", [
    ("Kot biegnie szybko", 1.0, pytest.approx(1 / 3)),
    ("Kot biegnie", 2.0, pytest.approx(1.0)),
    ("one two three four", 4.0, pytest.approx(1.0)),
    ("one two three four", 0.0, pytest.approx(0.0)),
])
def test_per_word_extract(text, counter_return, expected):
    extractor = PosTagPerWordExtractor(counter=_mock_counter(counter_return))
    assert extractor.extract(text) == expected


def test_per_word_zero_words_empty():
    spy = _mock_counter(5.0)
    extractor = PosTagPerWordExtractor(counter=spy)
    assert extractor.extract("") == 0.0
    spy.extract.assert_not_called()


def test_per_word_zero_words_whitespace():
    spy = _mock_counter(5.0)
    extractor = PosTagPerWordExtractor(counter=spy)
    assert extractor.extract("   ") == 0.0
    spy.extract.assert_not_called()


@pytest.mark.parametrize("tag, expected_name", [
    (PosTag.NOUN, "pos_noun_per_word"),
    (PosTag.VERB, "pos_verb_per_word"),
    (PosTag.ADJ, "pos_adj_per_word"),
    (PosTag.PUNCT, "pos_punct_per_word"),
])
def test_per_word_name_derives_from_counter_tag(tag, expected_name):
    extractor = PosTagPerWordExtractor(counter=PosTagCountExtractor(tag=tag))
    assert extractor.name == expected_name


def test_per_word_delegates_to_counter():
    spy = _mock_counter(2.0)
    extractor = PosTagPerWordExtractor(counter=spy)
    result = extractor.extract("Kot biegnie szybko czysto")
    spy.extract.assert_called_once_with("Kot biegnie szybko czysto")
    assert result == pytest.approx(0.5)
