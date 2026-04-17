import pytest
from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.punctuation.count import PunctuationCountExtractor
from ai_news_detector.features.punctuation.factory import PunctuationExtractorFactory
from ai_news_detector.features.punctuation.ratio import (
    PunctuationPerLetterExtractor,
    PunctuationPerWordExtractor,
)
from ai_news_detector.features.punctuation.variants import PunctuationVariant


@pytest.mark.parametrize("variant, expected_class", [
    (PunctuationVariant.COUNT, PunctuationCountExtractor),
    (PunctuationVariant.PER_WORD, PunctuationPerWordExtractor),
    (PunctuationVariant.PER_LETTER, PunctuationPerLetterExtractor),
])
def test_create_from_enum(variant, expected_class):
    extractor = PunctuationExtractorFactory.create(variant)
    assert isinstance(extractor, expected_class)


@pytest.mark.parametrize("variant_str, expected_class", [
    ("count", PunctuationCountExtractor),
    ("per_word", PunctuationPerWordExtractor),
    ("per_letter", PunctuationPerLetterExtractor),
])
def test_create_from_string(variant_str, expected_class):
    extractor = PunctuationExtractorFactory.create(variant_str)
    assert isinstance(extractor, expected_class)


@pytest.mark.parametrize("variant, expected_class", [
    (PunctuationVariant.COUNT, PunctuationCountExtractor),
    (PunctuationVariant.PER_WORD, PunctuationPerWordExtractor),
    (PunctuationVariant.PER_LETTER, PunctuationPerLetterExtractor),
])
def test_all_returned_objects_are_text_feature_extractors(variant, expected_class):
    extractor = PunctuationExtractorFactory.create(variant)
    assert isinstance(extractor, TextFeatureExtractor)


def test_create_unknown_string_raises_value_error():
    with pytest.raises(ValueError):
        PunctuationExtractorFactory.create("invalid_variant")
