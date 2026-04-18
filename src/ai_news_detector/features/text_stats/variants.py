from enum import StrEnum


class TextStatsVariant(StrEnum):
    TTR = "ttr"
    TTR_LEMMATIZED = "ttr_lemmatized"
    CAPITAL_RATIO = "capital_ratio"
    AVG_SENTENCE_LEN = "avg_sentence_len"
