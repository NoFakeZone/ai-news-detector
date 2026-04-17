import pytest
from unittest.mock import MagicMock
from ai_news_detector.features.punctuation.ratio import (
    PunctuationPerWordExtractor,
    PunctuationPerLetterExtractor,
)


@pytest.mark.parametrize("text, expected", [
    ("hi, there!", pytest.approx(1.0)),
    ("", 0.0),
    ("   ", 0.0),
    ("!!!", pytest.approx(3.0)),
    ("hello world", 0.0),
])
def test_per_word_extract(text, expected):
    assert PunctuationPerWordExtractor().extract(text) == expected


def test_per_word_name():
    assert PunctuationPerWordExtractor().name == "punctuation_per_word"


@pytest.mark.parametrize("text, expected", [
    ("hi!", pytest.approx(0.5)),
    ("", 0.0),
    ("!!!", 0.0),
    ("Hello!", pytest.approx(1 / 5)),
])
def test_per_letter_extract(text, expected):
    assert PunctuationPerLetterExtractor().extract(text) == expected


def test_per_letter_name():
    assert PunctuationPerLetterExtractor().name == "punctuation_per_letter"


def test_per_word_delegates_to_counter():
    spy = MagicMock()
    spy.extract.return_value = 4.0
    extractor = PunctuationPerWordExtractor(counter=spy)
    result = extractor.extract("one two")
    spy.extract.assert_called_once_with("one two")
    assert result == pytest.approx(2.0)


def test_per_letter_delegates_to_counter():
    spy = MagicMock()
    spy.extract.return_value = 2.0
    extractor = PunctuationPerLetterExtractor(counter=spy)
    result = extractor.extract("abcd")
    spy.extract.assert_called_once_with("abcd")
    assert result == pytest.approx(0.5)
