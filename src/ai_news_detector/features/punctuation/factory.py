from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.punctuation.variants import PunctuationVariant
from ai_news_detector.features.punctuation.count import PunctuationCountExtractor
from ai_news_detector.features.punctuation.ratio import (
    PunctuationPerWordExtractor,
    PunctuationPerLetterExtractor,
)


class PunctuationExtractorFactory:
    _registry: dict[PunctuationVariant, type[TextFeatureExtractor]] = {
        PunctuationVariant.COUNT: PunctuationCountExtractor,
        PunctuationVariant.PER_WORD: PunctuationPerWordExtractor,
        PunctuationVariant.PER_LETTER: PunctuationPerLetterExtractor,
    }

    @classmethod
    def create(cls, variant: PunctuationVariant | str) -> TextFeatureExtractor:
        key = PunctuationVariant(variant)
        return cls._registry[key]()
