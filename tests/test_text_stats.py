import pytest
from ai_news_detector.features.text_stats import (
    ttr,
    ttr_lemmatized,
    capital_ratio,
    avg_sentence_len,
)

try:
    import spacy as _spacy
    _has_model = _spacy.util.is_package("pl_core_news_sm")
except ImportError:
    _has_model = False

_requires_model = pytest.mark.skipif(not _has_model, reason="pl_core_news_sm not installed")


# --- ttr ---

@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("   ", 0.0),
    ("cat dog bird", pytest.approx(1.0)),
    ("cat cat cat", pytest.approx(1 / 3)),
    ("the cat sat on the mat", pytest.approx(5 / 6)),
])
def test_ttr(text, expected):
    assert ttr(text) == expected


def test_ttr_case_insensitive():
    assert ttr("Cat cat CAT") == pytest.approx(1 / 3)


def test_ttr_returns_float():
    assert isinstance(ttr("hello world"), float)


# --- ttr_lemmatized ---

def test_ttr_lemmatized_empty():
    assert ttr_lemmatized("") == 0.0


def test_ttr_lemmatized_whitespace():
    assert ttr_lemmatized("   ") == 0.0


def test_ttr_lemmatized_no_alpha_tokens():
    assert ttr_lemmatized("123 456", lemmatize=lambda t: []) == 0.0


def test_ttr_lemmatized_delegates_lowercased():
    received = []
    def capture(text):
        received.append(text)
        return ["kot", "biec", "szybko"]
    ttr_lemmatized("Kot Biegnie Szybko", lemmatize=capture)
    assert received == ["kot biegnie szybko"]


def test_ttr_lemmatized_all_unique():
    assert ttr_lemmatized("x", lemmatize=lambda t: ["kot", "biec", "szybko"]) == pytest.approx(1.0)


def test_ttr_lemmatized_repeated_lemmas():
    assert ttr_lemmatized("x", lemmatize=lambda t: ["kot", "kot", "pies"]) == pytest.approx(2 / 3)


def test_ttr_lemmatized_returns_float():
    assert isinstance(ttr_lemmatized("x", lemmatize=lambda t: ["a", "b"]), float)


@pytest.mark.slow
@_requires_model
def test_ttr_lemmatized_spacy_integration():
    result = ttr_lemmatized("Kot biegnie szybko. Kot śpi.")
    assert 0.0 < result <= 1.0


# --- capital_ratio ---

@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("all lowercase", 0.0),
    ("Hello world.", 0.0),
    ("Hello. World.", 0.0),
    ("Hello! World", 0.0),
    ("Hello? World", 0.0),
    ("Hello! World? Yes.", 0.0),
])
def test_capital_ratio_no_internal_caps(text, expected):
    assert capital_ratio(text) == pytest.approx(expected)


def test_capital_ratio_internal_caps():
    text = "hello WORLD"
    assert capital_ratio(text) == pytest.approx(5 / len(text))


def test_capital_ratio_returns_float():
    assert isinstance(capital_ratio("Hello"), float)


# --- avg_sentence_len ---

@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("   ", 0.0),
    ("one two three.", pytest.approx(3.0)),
    ("one two. three four five.", pytest.approx(2.5)),
    ("one two three", pytest.approx(3.0)),
])
def test_avg_sentence_len(text, expected):
    assert avg_sentence_len(text) == expected


def test_avg_sentence_len_returns_float():
    assert isinstance(avg_sentence_len("hello."), float)
