import re
import spacy #pip install spacy #python -m spacy download pl_core_news_sm
from enum import Enum

try:
    nlp = spacy.load("pl_core_news_sm", disable=['ner', 'parser'])
except OSError:
    print("Pobierz model komendą: python -m spacy download pl_core_news_sm")

class TextStatsVariant(Enum):
    TTR = "ttr"
    TTR_LEMATIZED = "ttr_lematized" 
    CAPITAL_RATIO = "capital_ratio"
    AVG_SENTENCE_LEN = "avg_sentence_len"

def get_ttr(text: str) -> float:
    """Klasyczny TTR (liczy każde odmienione słowo jako inne)"""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def get_ttr_lematized(text: str) -> float:
    """Zaawansowany TTR (sprowadza słowa do formy podstawowej przez spaCy)"""
    if not text.strip():
        return 0.0
    
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    
    if not lemmas:
        return 0.0
    return len(set(lemmas)) / len(lemmas)

def get_capital_ratio(text: str) -> float:
    """Mierzy stosunek wielkich liter (pomijając te na początku zdań)"""
    if not text:
        return 0.0
    all_caps = len(re.findall(r'[A-Z]', text))
    starts = len(re.findall(r'(^|[.;]\s*)[A-Z]', text))
    internal_caps = max(0, all_caps - starts)
    return internal_caps / len(text)

def get_avg_sentence_len(text: str) -> float:
    """Mierzy średnią liczbę słów w zdaniu"""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    return sum(word_counts) / len(sentences)

FEATURE_MAP = {
    TextStatsVariant.TTR: get_ttr,
    TextStatsVariant.TTR_LEMATIZED: get_ttr_lematized,
    TextStatsVariant.CAPITAL_RATIO: get_capital_ratio,
    TextStatsVariant.AVG_SENTENCE_LEN: get_avg_sentence_len,
}