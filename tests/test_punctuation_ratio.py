import pytest

from ai_news_detector.features.punctuation import (
    punctuation_per_word,
    punctuation_per_letter,
)


@pytest.mark.parametrize("text, expected", [
    ("hi, there!", pytest.approx(1.0)),
    ("", 0.0),
    ("   ", 0.0),
    ("!!!", pytest.approx(3.0)),
    ("hello world", 0.0),
])
def test_punctuation_per_word(text, expected):
    assert punctuation_per_word(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("hi!", pytest.approx(0.5)),
    ("", 0.0),
    ("!!!", 0.0),
    ("Hello!", pytest.approx(1 / 5)),
])
def test_punctuation_per_letter(text, expected):
    assert punctuation_per_letter(text) == expected
