from collections.abc import Callable

from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.pos.count import PosTagCountExtractor
from ai_news_detector.features.pos.ratio import PosTagPerWordExtractor
from ai_news_detector.features.pos.tags import PosTag
from ai_news_detector.features.pos.variants import PosVariant


class PosExtractorFactory:
    _registry: dict[PosVariant, Callable[[PosTag], TextFeatureExtractor]] = {
        PosVariant.COUNT: lambda t: PosTagCountExtractor(tag=t),
        PosVariant.PER_WORD: lambda t: PosTagPerWordExtractor(counter=PosTagCountExtractor(tag=t)),
    }

    @classmethod
    def create(
        cls,
        variant: PosVariant | str,
        tag: PosTag | str = PosTag.NOUN,
    ) -> TextFeatureExtractor:
        return cls._registry[PosVariant(variant)](PosTag(tag))
