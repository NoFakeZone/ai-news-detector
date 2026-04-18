from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.text_stats.variants import TextStatsVariant
from ai_news_detector.features.text_stats.extractors import (
    TtrExtractor,
    TtrLemmatizedExtractor,
    CapitalRatioExtractor,
    AvgSentenceLenExtractor,
)


class TextStatsExtractorFactory:
    _registry: dict[TextStatsVariant, type[TextFeatureExtractor]] = {
        TextStatsVariant.TTR: TtrExtractor,
        TextStatsVariant.TTR_LEMMATIZED: TtrLemmatizedExtractor,
        TextStatsVariant.CAPITAL_RATIO: CapitalRatioExtractor,
        TextStatsVariant.AVG_SENTENCE_LEN: AvgSentenceLenExtractor,
    }

    @classmethod
    def create(cls, variant: TextStatsVariant | str) -> TextFeatureExtractor:
        return cls._registry[TextStatsVariant(variant)]()
