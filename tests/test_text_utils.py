import pytest
from ai_news_detector.features.text_utils import count_words, count_letters


@pytest.mark.parametrize("text, expected", [
    ("", 0),
    ("   ", 0),
    ("hello", 1),
    ("hello world", 2),
    ("  hello   world  ", 2),
])
def test_count_words(text, expected):
    assert count_words(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("", 0),
    ("hello", 5),
    ("hi!", 2),
    ("żółć", 4),
    ("123", 0),
    ("!!!", 0),
])
def test_count_letters(text, expected):
    assert count_letters(text) == expected
