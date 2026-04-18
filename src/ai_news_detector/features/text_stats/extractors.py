import re
from dataclasses import dataclass, field

from ai_news_detector.features.base import TextFeatureExtractor
from ai_news_detector.features.text_stats.lemmatizer import Lemmatizer, SpacyPolishLemmatizer

_WORD_RE = re.compile(r'\b\w+\b')
_SENTENCE_SEP_RE = re.compile(r'[.!?]+')


@dataclass(frozen=True)
class TtrExtractor(TextFeatureExtractor):
    @property
    def name(self) -> str:
        return "ttr"

    def extract(self, text: str) -> float:
        words = _WORD_RE.findall(text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)


@dataclass(frozen=True)
class TtrLemmatizedExtractor(TextFeatureExtractor):
    lemmatizer: Lemmatizer = field(default_factory=SpacyPolishLemmatizer)

    @property
    def name(self) -> str:
        return "ttr_lemmatized"

    def extract(self, text: str) -> float:
        if not text.strip():
            return 0.0
        lemmas = self.lemmatizer.lemmatize(text.lower())
        if not lemmas:
            return 0.0
        return len(set(lemmas)) / len(lemmas)


@dataclass(frozen=True)
class CapitalRatioExtractor(TextFeatureExtractor):
    @property
    def name(self) -> str:
        return "capital_ratio"

    def extract(self, text: str) -> float:
        if not text:
            return 0.0
        all_caps = len(re.findall(r'[A-Z]', text))
        sentence_starts = len(re.findall(r'(?:^|[.;]\s*)[A-Z]', text))
        internal_caps = max(0, all_caps - sentence_starts)
        return internal_caps / len(text)


@dataclass(frozen=True)
class AvgSentenceLenExtractor(TextFeatureExtractor):
    @property
    def name(self) -> str:
        return "avg_sentence_len"

    def extract(self, text: str) -> float:
        sentences = [s.strip() for s in _SENTENCE_SEP_RE.split(text) if s.strip()]
        if not sentences:
            return 0.0
        word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
        return sum(word_counts) / len(sentences)
