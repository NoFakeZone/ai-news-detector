"""Small demo exercising the punctuation and POS extractors on sample texts."""
from ai_news_detector.features.punctuation import (
    punctuation_count,
    punctuation_per_word,
    punctuation_per_letter,
)
from ai_news_detector.features.pos import pos_count, pos_per_word


PUNCT_FNS = [
    ("punctuation_count", punctuation_count),
    ("punctuation_per_word", punctuation_per_word),
    ("punctuation_per_letter", punctuation_per_letter),
]
POS_FNS = [("count", pos_count), ("per_word", pos_per_word)]
TAGS = ["NOUN", "VERB", "ADJ"]


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
        for tag in TAGS:
            for variant_name, fn in POS_FNS:
                label = f"pos_{tag.lower()}_{variant_name}"
                print(f"{label:>28}: {fn(text, tag):.4f}")
    except RuntimeError as exc:
        print(f"[skipped] {exc}")


def main() -> None:
    _demo_punctuation()
    _demo_pos()


if __name__ == "__main__":
    main()
