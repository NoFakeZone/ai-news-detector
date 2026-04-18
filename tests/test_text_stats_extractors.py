import pytest
from unittest.mock import MagicMock

from ai_news_detector.features.text_stats import (
    TtrExtractor,
    TtrLemmatizedExtractor,
    CapitalRatioExtractor,
    AvgSentenceLenExtractor,
)


def _mock_lemmatizer(lemmas: list[str]) -> MagicMock:
    m = MagicMock()
    m.lemmatize.return_value = lemmas
    return m


# --- TtrExtractor ---

class TestTtrExtractor:
    def test_name(self):
        assert TtrExtractor().name == "ttr"

    def test_returns_float(self):
        assert isinstance(TtrExtractor().extract("hello world"), float)

    def test_all_unique(self):
        assert TtrExtractor().extract("cat dog bird") == pytest.approx(1.0)

    def test_all_repeated(self):
        assert TtrExtractor().extract("cat cat cat") == pytest.approx(1 / 3)

    def test_mixed(self):
        assert TtrExtractor().extract("the cat sat on the mat") == pytest.approx(5 / 6)

    def test_empty(self):
        assert TtrExtractor().extract("") == 0.0

    def test_whitespace_only(self):
        assert TtrExtractor().extract("   ") == 0.0

    def test_case_insensitive(self):
        assert TtrExtractor().extract("Cat cat CAT") == pytest.approx(1 / 3)


# --- TtrLemmatizedExtractor ---

class TestTtrLemmatizedExtractor:
    def test_name(self):
        assert TtrLemmatizedExtractor().name == "ttr_lemmatized"

    def test_delegates_to_lemmatizer(self):
        mock = _mock_lemmatizer(["kot", "biec", "szybko"])
        TtrLemmatizedExtractor(lemmatizer=mock).extract("Kot biegnie szybko")
        mock.lemmatize.assert_called_once_with("kot biegnie szybko")

    def test_all_unique(self):
        mock = _mock_lemmatizer(["kot", "biec", "szybko"])
        result = TtrLemmatizedExtractor(lemmatizer=mock).extract("text")
        assert result == pytest.approx(1.0)

    def test_repeated_lemmas(self):
        mock = _mock_lemmatizer(["kot", "kot", "pies"])
        result = TtrLemmatizedExtractor(lemmatizer=mock).extract("text")
        assert result == pytest.approx(2 / 3)

    def test_empty_text(self):
        mock = _mock_lemmatizer([])
        result = TtrLemmatizedExtractor(lemmatizer=mock).extract("")
        assert result == 0.0
        mock.lemmatize.assert_not_called()

    def test_whitespace_only(self):
        mock = _mock_lemmatizer([])
        result = TtrLemmatizedExtractor(lemmatizer=mock).extract("   ")
        assert result == 0.0
        mock.lemmatize.assert_not_called()

    def test_no_alpha_tokens(self):
        mock = _mock_lemmatizer([])
        result = TtrLemmatizedExtractor(lemmatizer=mock).extract("123 456")
        assert result == 0.0

    def test_returns_float(self):
        mock = _mock_lemmatizer(["a", "b"])
        assert isinstance(TtrLemmatizedExtractor(lemmatizer=mock).extract("a b"), float)


# --- CapitalRatioExtractor ---

class TestCapitalRatioExtractor:
    def test_name(self):
        assert CapitalRatioExtractor().name == "capital_ratio"

    def test_returns_float(self):
        assert isinstance(CapitalRatioExtractor().extract("Hello"), float)

    def test_empty(self):
        assert CapitalRatioExtractor().extract("") == 0.0

    def test_no_internal_caps(self):
        assert CapitalRatioExtractor().extract("Hello world.") == pytest.approx(0.0)

    def test_internal_caps_counted(self):
        text = "hello WORLD"
        result = CapitalRatioExtractor().extract(text)
        assert result == pytest.approx(5 / len(text))

    def test_sentence_start_not_counted(self):
        text = "Hello. World."
        result = CapitalRatioExtractor().extract(text)
        assert result == pytest.approx(0.0)

    def test_lowercase_only(self):
        assert CapitalRatioExtractor().extract("all lowercase") == pytest.approx(0.0)


# --- AvgSentenceLenExtractor ---

class TestAvgSentenceLenExtractor:
    def test_name(self):
        assert AvgSentenceLenExtractor().name == "avg_sentence_len"

    def test_returns_float(self):
        assert isinstance(AvgSentenceLenExtractor().extract("Hello world."), float)

    def test_empty(self):
        assert AvgSentenceLenExtractor().extract("") == 0.0

    def test_whitespace_only(self):
        assert AvgSentenceLenExtractor().extract("   ") == 0.0

    def test_single_sentence(self):
        assert AvgSentenceLenExtractor().extract("one two three.") == pytest.approx(3.0)

    def test_multiple_sentences(self):
        result = AvgSentenceLenExtractor().extract("one two. three four five.")
        assert result == pytest.approx(2.5)

    def test_question_and_exclamation(self):
        # "Hello" (1), "Yes" (1), "Of course" (2) → avg = 4/3
        result = AvgSentenceLenExtractor().extract("Hello? Yes! Of course.")
        assert result == pytest.approx(4 / 3)

    def test_no_punctuation(self):
        result = AvgSentenceLenExtractor().extract("one two three")
        assert result == pytest.approx(3.0)
