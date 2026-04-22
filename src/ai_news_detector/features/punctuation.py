import string

from ai_news_detector.features.text_utils import count_words, count_letters

DEFAULT_PUNCTUATION = frozenset(string.punctuation)


def punctuation_count(text: str, chars: frozenset[str] = DEFAULT_PUNCTUATION) -> float:
    return float(sum(1 for c in text if c in chars))


def punctuation_per_word(text: str, chars: frozenset[str] = DEFAULT_PUNCTUATION) -> float:
    words = count_words(text)
    if words == 0:
        return 0.0
    return punctuation_count(text, chars) / words


def punctuation_per_letter(text: str, chars: frozenset[str] = DEFAULT_PUNCTUATION) -> float:
    letters = count_letters(text)
    if letters == 0:
        return 0.0
    return punctuation_count(text, chars) / letters
