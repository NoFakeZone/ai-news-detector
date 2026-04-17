from dataclasses import dataclass, field

from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.text_utils import count_words, count_letters
from ai_news_detector.features.punctuation.count import PunctuationCountExtractor


@dataclass(frozen=True)
class PunctuationPerWordExtractor(TextFeatureExtractor):
    counter: PunctuationCountExtractor = field(default_factory=PunctuationCountExtractor)

    @property
    def name(self) -> str:
        return "punctuation_per_word"

    def extract(self, text: str) -> float:
        words = count_words(text)
        if words == 0:
            return 0.0
        return self.counter.extract(text) / words


@dataclass(frozen=True)
class PunctuationPerLetterExtractor(TextFeatureExtractor):
    counter: PunctuationCountExtractor = field(default_factory=PunctuationCountExtractor)

    @property
    def name(self) -> str:
        return "punctuation_per_letter"

    def extract(self, text: str) -> float:
        letters = count_letters(text)
        if letters == 0:
            return 0.0
        return self.counter.extract(text) / letters
