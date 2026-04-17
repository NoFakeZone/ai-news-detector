"""Small demo exercising the punctuation and POS extractors on sample texts."""
from ai_news_detector.features.pos import (
    PosExtractorFactory,
    PosTag,
    PosVariant,
)
from ai_news_detector.features.punctuation import (
    PunctuationExtractorFactory,
    PunctuationVariant,
)


def _demo_punctuation() -> None:
    text = "Hello, world! How are you?"
    print("Punctuation features")
    print(f"text: {text!r}\n")

    for variant in PunctuationVariant:
        extractor = PunctuationExtractorFactory.create(variant)
        print(f"{extractor.name:>28}: {extractor.extract(text):.4f}")


def _demo_pos() -> None:
    text = "Kot biegnie szybko przez zielony ogród."
    print("\nPOS features (Polish)")
    print(f"text: {text!r}\n")

    try:
        for tag in (PosTag.NOUN, PosTag.VERB, PosTag.ADJ):
            for variant in PosVariant:
                extractor = PosExtractorFactory.create(variant, tag)
                print(f"{extractor.name:>28}: {extractor.extract(text):.4f}")
    except RuntimeError as exc:
        print(f"[skipped] {exc}")


def main() -> None:
    _demo_punctuation()
    _demo_pos()


if __name__ == "__main__":
    main()
