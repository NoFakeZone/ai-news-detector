"""Small demo exercising the punctuation, POS, and text stats feature functions on sample texts."""
from ai_news_detector.features.punctuation import (
    punctuation_count,
    punctuation_per_word,
    punctuation_per_letter,
)
from ai_news_detector.features.pos import pos_count, pos_per_word
from ai_news_detector.features.text_stats import (
    ttr,
    ttr_lemmatized,
    capital_ratio,
    avg_sentence_len,
)

PUNCT_FNS = [
    ("punctuation_count", punctuation_count),
    ("punctuation_per_word", punctuation_per_word),
    ("punctuation_per_letter", punctuation_per_letter),
]
POS_TAGS = ["NOUN", "VERB", "ADJ"]


def _demo_punctuation() -> None:
    text = "Hello, world! How are you?"
    print("Punctuation features")
    print(f"text: {text!r}\n")
    for name, fn in PUNCT_FNS:
        print(f"{name:>28}: {fn(text):.4f}")


def _demo_pos() -> None:
    text = "Kot biegnie szybko przez zielony ogród."
    print("\nPOS features (Polish)")
    print(f"text: {text!r}\n")
    try:
        for tag in POS_TAGS:
            print(f"{'pos_' + tag.lower() + '_count':>28}: {pos_count(text, tag):.4f}")
            print(f"{'pos_' + tag.lower() + '_per_word':>28}: {pos_per_word(text, tag):.4f}")
    except RuntimeError as exc:
        print(f"[skipped] {exc}")


def _demo_text_stats() -> None:
    text = (
        "Sztuczne sieci neuronowe stanowią fundament współczesnej informatyki. "
        "Inspiracją była budowa ludzkiego mózgu. "
        "Programiści muszą dbać o optymalizację algorytmów, aby sieci działały wydajniej."
    )
    print("\nText stats features (Polish)")
    print(f"text: {text!r}\n")
    print(f"{'ttr':>28}: {ttr(text):.4f}")
    print(f"{'capital_ratio':>28}: {capital_ratio(text):.4f}")
    print(f"{'avg_sentence_len':>28}: {avg_sentence_len(text):.4f}")
    try:
        print(f"{'ttr_lemmatized':>28}: {ttr_lemmatized(text):.4f}")
    except RuntimeError as exc:
        print(f"{'ttr_lemmatized':>28}: [skipped] {exc}")


def main() -> None:
    _demo_punctuation()
    _demo_pos()
    _demo_text_stats()


if __name__ == "__main__":
    main()
