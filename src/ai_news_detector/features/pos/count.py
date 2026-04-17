from dataclasses import dataclass

from ai_news_detector.features.pos.base import PosExtractor


@dataclass(frozen=True)
class PosTagCountExtractor(PosExtractor):
    @property
    def name(self) -> str:
        return f"pos_{self.tag.value.lower()}_count"

    def extract(self, text: str) -> float:
        if not text.strip():
            return 0.0
        tagged = self.tagger.tag(text)
        return float(sum(1 for _, t in tagged if t == self.tag.value))
