import string
from dataclasses import dataclass, field

from ai_news_detector.features.base import TextFeatureExtractor


@dataclass(frozen=True)
class PunctuationExtractor(TextFeatureExtractor):
    punctuation_chars: frozenset[str] = field(default_factory=lambda: frozenset(string.punctuation))
