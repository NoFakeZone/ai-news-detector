import pytest
from unittest.mock import MagicMock

from ai_news_detector.features.pos import pos_count


def _mock_tagger(tagged):
    m = MagicMock()
    m.return_value = tagged
    return m


@pytest.mark.parametrize("tagged, target_tag, expected", [
    ([("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], "NOUN", 1.0),
    ([("Kot", "NOUN"), ("biegnie", "VERB"), ("szybko", "ADV")], "VERB", 1.0),
    ([("Kot", "NOUN"), ("pies", "NOUN"), ("biegnie", "VERB")], "NOUN", 2.0),
    ([("Kot", "NOUN"), ("biegnie", "VERB")], "ADJ", 0.0),
    ([], "NOUN", 0.0),
    ([("a", "ADJ"), ("b", "ADJ"), ("c", "ADJ")], "ADJ", 3.0),
])
def test_pos_count_counts_matching_tag(tagged, target_tag, expected):
    assert pos_count("irrelevant", tag=target_tag, tagger=_mock_tagger(tagged)) == expected


def test_pos_count_empty_text():
    spy = _mock_tagger([])
    assert pos_count("", tagger=spy) == 0.0
    spy.assert_not_called()


def test_pos_count_whitespace_only():
    spy = _mock_tagger([])
    assert pos_count("   \t\n  ", tagger=spy) == 0.0
    spy.assert_not_called()


def test_pos_count_returns_float():
    result = pos_count("Kot", tag="NOUN", tagger=_mock_tagger([("Kot", "NOUN")]))
    assert isinstance(result, float)


def test_pos_count_tagger_called_once_with_text():
    spy = _mock_tagger([("Kot", "NOUN"), ("biegnie", "VERB")])
    pos_count("Kot biegnie", tag="NOUN", tagger=spy)
    spy.assert_called_once_with("Kot biegnie")


def test_pos_count_validate_raises_on_unknown_tag():
    with pytest.raises(ValueError, match="NOU N"):
        pos_count("Kot", tag="NOU N", tagger=_mock_tagger([]), validate=True)


def test_pos_count_validate_accepts_known_tag():
    result = pos_count("Kot", tag="NOUN", tagger=_mock_tagger([("Kot", "NOUN")]), validate=True)
    assert result == 1.0


def test_pos_count_no_validate_silent_on_unknown_tag():
    result = pos_count("Kot", tag="NOU N", tagger=_mock_tagger([("Kot", "NOUN")]))
    assert result == 0.0
