from dataclasses import dataclass, field

from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.pos.tagger import PosTagger, SpacyPolishPosTagger
from ai_news_detector.features.pos.tags import PosTag


@dataclass(frozen=True)
class PosExtractor(TextFeatureExtractor):
    tag: PosTag = PosTag.NOUN
    tagger: PosTagger = field(default_factory=SpacyPolishPosTagger)
