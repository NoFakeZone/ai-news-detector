"""Small demo exercising the punctuation, POS, and text stats feature extractors on sample texts."""
from ai_news_detector.features.punctuation import (
    PunctuationExtractorFactory,
    PunctuationVariant,
)
from ai_news_detector.features.pos import (
    PosExtractorFactory,
    PosTag,
    PosVariant,
)
from ai_news_detector.features.text_stats import (
    TextStatsExtractorFactory,
    TextStatsVariant,
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
        for tag in [PosTag.NOUN, PosTag.VERB, PosTag.ADJ]:
            for variant in PosVariant:
                extractor = PosExtractorFactory.create(variant, tag)
                print(f"{extractor.name:>28}: {extractor.extract(text):.4f}")
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

    for variant in TextStatsVariant:
        extractor = TextStatsExtractorFactory.create(variant)
        try:
            value = extractor.extract(text)
            print(f"{extractor.name:>28}: {value:.4f}")
        except RuntimeError as exc:
            print(f"{extractor.name:>28}: [skipped] {exc}")


def main() -> None:
    _demo_punctuation()
    _demo_pos()
    _demo_text_stats()


if __name__ == "__main__":
    main()
