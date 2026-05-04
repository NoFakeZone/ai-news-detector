import pytest

from ai_news_detector.features.syllables import (
    avg_syllables_per_sentence,
    avg_word_length,
    count_syllables_word,
)


@pytest.mark.parametrize(
    "word, expected",
    [
        ("kot", 1),
        ("żółć", 1),
        ("dzięki", 2),
        ("", 0),
        ("123", 0),
    ],
)
def test_count_syllables_word(word, expected):
    assert count_syllables_word(word) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0.0),
        ("   ", 0.0),
        ("Kot pies.", pytest.approx(2.0)),
        ("Kot pies. Kot pies.", pytest.approx(2.0)),
        ("Aa. Bee.", pytest.approx(1.0)),
        ("Jedna. Druga większa.", pytest.approx((2.0 + 4.0) / 2)),
    ],
)
def test_avg_syllables_per_sentence(text, expected):
    assert avg_syllables_per_sentence(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0.0),
        ("kot pies", pytest.approx(3.5)),
        ("żółć", pytest.approx(4.0)),
    ],
)
def test_avg_word_length(text, expected):
    assert avg_word_length(text) == expected
