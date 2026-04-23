import pytest
from ai_news_detector.features.punctuation import (
    punctuation_count,
    punctuation_per_word,
    punctuation_per_letter,
    DEFAULT_PUNCTUATION,
)


@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("hello", 0.0),
    ("hi!", 1.0),
    ("a, b; c.", 3.0),
    ("!!!", 3.0),
    ("Hello, world! How are you?", 3.0),
])
def test_count(text, expected):
    assert punctuation_count(text) == expected


def test_count_returns_float():
    assert isinstance(punctuation_count("hi!"), float)


def test_count_custom_chars():
    assert punctuation_count("hi!, bye.", chars=frozenset({"!"})) == 1.0


@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("   ", 0.0),
    ("hi, there!", pytest.approx(1.0)),
    ("!!!", pytest.approx(3.0)),
    ("hello world", 0.0),
])
def test_per_word(text, expected):
    assert punctuation_per_word(text) == expected


def test_per_word_returns_float():
    assert isinstance(punctuation_per_word("hi,"), float)


@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("!!!", 0.0),
    ("hi!", pytest.approx(0.5)),
    ("Hello!", pytest.approx(1 / 5)),
])
def test_per_letter(text, expected):
    assert punctuation_per_letter(text) == expected


def test_per_letter_returns_float():
    assert isinstance(punctuation_per_letter("hi!"), float)


def test_default_punctuation_is_frozenset():
    assert isinstance(DEFAULT_PUNCTUATION, frozenset)
    assert "!" in DEFAULT_PUNCTUATION
