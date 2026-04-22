def count_words(text: str) -> int:
    return len(text.split())


def count_letters(text: str) -> int:
    return sum(1 for c in text if c.isalpha())
