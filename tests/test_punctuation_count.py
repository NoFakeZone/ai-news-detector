import pytest

from ai_news_detector.features.punctuation import punctuation_count


@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("hello", 0.0),
    ("hi!", 1.0),
    ("a, b; c.", 3.0),
    ("!!!", 3.0),
    ("Hello, world! How are you?", 3.0),
])
def test_punctuation_count(text, expected):
    assert punctuation_count(text) == expected


def test_custom_chars():
    assert punctuation_count("hi!, bye.", chars=frozenset({"!"})) == 1.0


def test_returns_float():
    assert isinstance(punctuation_count("hi!"), float)
