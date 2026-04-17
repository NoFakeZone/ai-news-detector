from dataclasses import dataclass

from ai_news_detector.features.punctuation.base import PunctuationExtractor


@dataclass(frozen=True)
class PunctuationCountExtractor(PunctuationExtractor):
    @property
    def name(self) -> str:
        return "punctuation_count"

    def extract(self, text: str) -> float:
        return float(sum(1 for c in text if c in self.punctuation_chars))
