"""Small demo exercising the punctuation extractors on a sample text."""
from ai_news_detector.features.punctuation import (
    PunctuationExtractorFactory,
    PunctuationVariant,
)


def main() -> None:
    text = "Hello, world! How are you?"
    print(f"text: {text!r}\n")

    for variant in PunctuationVariant:
        extractor = PunctuationExtractorFactory.create(variant)
        print(f"{extractor.name:>24}: {extractor.extract(text):.4f}")


if __name__ == "__main__":
    main()
