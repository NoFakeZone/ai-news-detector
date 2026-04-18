import pytest
from unittest.mock import MagicMock

from ai_news_detector.features.pos import pos_per_word


def _mock_tagger(tagged):
    m = MagicMock()
    m.return_value = tagged
    return m


@pytest.mark.parametrize("text, tagged, expected", [
    ("Kot biegnie szybko", [("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], pytest.approx(1 / 3)),
    ("Kot pies", [("Kot", "NOUN"), ("pies", "NOUN")], pytest.approx(1.0)),
    ("one two three four", [("one", "NUM"), ("two", "NUM"), ("three", "NUM"), ("four", "NUM")], pytest.approx(0.0)),
    ("a b c d", [("a", "NOUN"), ("b", "NOUN"), ("c", "NOUN"), ("d", "NOUN")], pytest.approx(1.0)),
])
def test_pos_per_word(text, tagged, expected):
    assert pos_per_word(text, tag="NOUN", tagger=_mock_tagger(tagged)) == expected


def test_pos_per_word_empty():
    spy = _mock_tagger([])
    assert pos_per_word("", tagger=spy) == 0.0
    spy.assert_not_called()


def test_pos_per_word_whitespace():
    spy = _mock_tagger([])
    assert pos_per_word("   ", tagger=spy) == 0.0
    spy.assert_not_called()


def test_pos_per_word_passes_tagger_through():
    spy = _mock_tagger([("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV"), ("czysto", "ADV")])
    result = pos_per_word("Kot biegnie szybko czysto", tag="NOUN", tagger=spy)
    spy.assert_called_once_with("Kot biegnie szybko czysto")
    assert result == pytest.approx(0.25)


def test_pos_per_word_validate_raises_on_unknown_tag():
    with pytest.raises(ValueError, match="NOU N"):
        pos_per_word("Kot", tag="NOU N", tagger=_mock_tagger([]), validate=True)


def test_pos_per_word_validate_accepts_known_tag():
    result = pos_per_word("Kot biegnie", tag="NOUN", tagger=_mock_tagger([("Kot", "NOUN"), ("biegnie", "VERB")]), validate=True)
    assert result == pytest.approx(0.5)


def test_pos_per_word_no_validate_silent_on_unknown_tag():
    result = pos_per_word("Kot", tag="NOU N", tagger=_mock_tagger([("Kot", "NOUN")]))
    assert result == 0.0
