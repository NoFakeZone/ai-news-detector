import pytest

from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.pos import (
    PosExtractorFactory,
    PosTag,
    PosTagCountExtractor,
    PosTagPerWordExtractor,
    PosVariant,
)


@pytest.mark.parametrize("variant, tag, expected_class", [
    (PosVariant.COUNT, PosTag.NOUN, PosTagCountExtractor),
    (PosVariant.COUNT, PosTag.VERB, PosTagCountExtractor),
    (PosVariant.PER_WORD, PosTag.NOUN, PosTagPerWordExtractor),
    (PosVariant.PER_WORD, PosTag.VERB, PosTagPerWordExtractor),
])
def test_create_from_enum(variant, tag, expected_class):
    extractor = PosExtractorFactory.create(variant, tag)
    assert isinstance(extractor, expected_class)


@pytest.mark.parametrize("variant_str, tag_str, expected_class", [
    ("count", "NOUN", PosTagCountExtractor),
    ("count", "VERB", PosTagCountExtractor),
    ("per_word", "NOUN", PosTagPerWordExtractor),
    ("per_word", "VERB", PosTagPerWordExtractor),
])
def test_create_from_string(variant_str, tag_str, expected_class):
    extractor = PosExtractorFactory.create(variant_str, tag_str)
    assert isinstance(extractor, expected_class)


def test_count_has_expected_tag():
    extractor = PosExtractorFactory.create(PosVariant.COUNT, PosTag.VERB)
    assert extractor.tag == PosTag.VERB


def test_ratio_wires_counter_with_tag():
    extractor = PosExtractorFactory.create(PosVariant.PER_WORD, PosTag.VERB)
    assert isinstance(extractor, PosTagPerWordExtractor)
    assert extractor.counter.tag == PosTag.VERB


@pytest.mark.parametrize("variant, tag", [
    (PosVariant.COUNT, PosTag.NOUN),
    (PosVariant.COUNT, PosTag.VERB),
    (PosVariant.PER_WORD, PosTag.NOUN),
    (PosVariant.PER_WORD, PosTag.VERB),
])
def test_returned_are_text_feature_extractors(variant, tag):
    extractor = PosExtractorFactory.create(variant, tag)
    assert isinstance(extractor, TextFeatureExtractor)


def test_unknown_variant_raises_value_error():
    with pytest.raises(ValueError):
        PosExtractorFactory.create("bogus")


def test_unknown_tag_raises_value_error():
    with pytest.raises(ValueError):
        PosExtractorFactory.create("count", "BOGUS")


def test_default_tag_is_noun():
    extractor = PosExtractorFactory.create(PosVariant.COUNT)
    assert extractor.tag == PosTag.NOUN
