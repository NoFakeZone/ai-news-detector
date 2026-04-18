import pytest

from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.text_stats import (
    TextStatsVariant,
    TextStatsExtractorFactory,
    TtrExtractor,
    TtrLemmatizedExtractor,
    CapitalRatioExtractor,
    AvgSentenceLenExtractor,
)


@pytest.mark.parametrize("variant, expected_class", [
    (TextStatsVariant.TTR, TtrExtractor),
    (TextStatsVariant.TTR_LEMMATIZED, TtrLemmatizedExtractor),
    (TextStatsVariant.CAPITAL_RATIO, CapitalRatioExtractor),
    (TextStatsVariant.AVG_SENTENCE_LEN, AvgSentenceLenExtractor),
])
def test_create_from_enum(variant, expected_class):
    extractor = TextStatsExtractorFactory.create(variant)
    assert isinstance(extractor, expected_class)


@pytest.mark.parametrize("variant_str, expected_class", [
    ("ttr", TtrExtractor),
    ("ttr_lemmatized", TtrLemmatizedExtractor),
    ("capital_ratio", CapitalRatioExtractor),
    ("avg_sentence_len", AvgSentenceLenExtractor),
])
def test_create_from_string(variant_str, expected_class):
    extractor = TextStatsExtractorFactory.create(variant_str)
    assert isinstance(extractor, expected_class)


@pytest.mark.parametrize("variant", list(TextStatsVariant))
def test_returned_are_text_feature_extractors(variant):
    assert isinstance(TextStatsExtractorFactory.create(variant), TextFeatureExtractor)


def test_unknown_variant_raises_value_error():
    with pytest.raises(ValueError):
        TextStatsExtractorFactory.create("bogus")
