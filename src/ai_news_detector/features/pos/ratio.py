from dataclasses import dataclass, field

from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.pos.count import PosTagCountExtractor
from ai_news_detector.features.text_utils import count_words


@dataclass(frozen=True)
class PosTagPerWordExtractor(TextFeatureExtractor):
    counter: PosTagCountExtractor = field(default_factory=PosTagCountExtractor)

    @property
    def name(self) -> str:
        return f"pos_{self.counter.tag.value.lower()}_per_word"

    def extract(self, text: str) -> float:
        words = count_words(text)
        if words == 0:
            return 0.0
        return self.counter.extract(text) / words
