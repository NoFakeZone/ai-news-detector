import pytest
from ai_news_detector.features.punctuation.count import PunctuationCountExtractor


@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("hello", 0.0),
    ("hi!", 1.0),
    ("a, b; c.", 3.0),
    ("!!!", 3.0),
    ("Hello, world! How are you?", 3.0),
])
def test_extract(text, expected):
    assert PunctuationCountExtractor().extract(text) == expected


def test_name():
    assert PunctuationCountExtractor().name == "punctuation_count"


def test_custom_punctuation_chars():
    extractor = PunctuationCountExtractor(punctuation_chars=frozenset({"!"}))
    assert extractor.extract("hi!, bye.") == 1.0


def test_extract_returns_float():
    result = PunctuationCountExtractor().extract("hi!")
    assert isinstance(result, float)
