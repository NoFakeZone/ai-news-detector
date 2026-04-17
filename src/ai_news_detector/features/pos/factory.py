from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.pos.count import PosTagCountExtractor
from ai_news_detector.features.pos.ratio import PosTagPerWordExtractor
from ai_news_detector.features.pos.tags import PosTag
from ai_news_detector.features.pos.variants import PosVariant


class PosExtractorFactory:
    _registry: dict[PosVariant, type[TextFeatureExtractor]] = {
        PosVariant.COUNT: PosTagCountExtractor,
        PosVariant.PER_WORD: PosTagPerWordExtractor,
    }

    @classmethod
    def create(
        cls,
        variant: PosVariant | str,
        tag: PosTag | str = PosTag.NOUN,
    ) -> TextFeatureExtractor:
        v = PosVariant(variant)
        t = PosTag(tag)
        if v is PosVariant.COUNT:
            return PosTagCountExtractor(tag=t)
        return PosTagPerWordExtractor(counter=PosTagCountExtractor(tag=t))
