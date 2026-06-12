from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DM_DIR = Path(__file__).resolve().parent
REPO_ROOT = DM_DIR.parent
SRC_DIR = REPO_ROOT / "src"
BENCHMARKS_PATH = DM_DIR / "eda_benchmarks.json"
BASIC_DICT_CACHE = DM_DIR / "basic_popularity_dict.json"
WIKI_DICT_CANDIDATES = (
    REPO_ROOT / "wiki_popularity_dict.json",
    SRC_DIR / "wiki_popularity_dict.json",
    DM_DIR / "wiki_popularity_dict.json",
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_news_detector.features.pos import UD_TAGS, all_pos_per_word
from ai_news_detector.features.text_stats import avg_sentence_len, ttr_lemmatized
from load_dataset import build_popularity_dictionary, preprocess_for_bert

MIN_WORDS = 15


@dataclass(frozen=True)
class SignalResult:
    signal_id: str
    label: str
    description: str
    value: float
    ai_score: float
    weight: float
    unit: str
    human_center: float | None = None
    ai_center: float | None = None
    chart_path: Path | None = None


@dataclass(frozen=True)
class AnalysisResult:
    label: str
    probability_ai: float
    probability_human: float
    word_count: int
    signals: list[SignalResult]
    reference_signals: list[SignalResult]
    warning: str | None = None


def _find_chart(keyword: str | None, exclude: tuple[str, ...] = ()) -> Path | None:
    if not keyword:
        return None
    matches = []
    for path in DM_DIR.glob("*.png"):
        name = path.name.casefold()
        if keyword.casefold() not in name:
            continue
        if any(token.casefold() in name for token in exclude):
            continue
        matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda path: len(path.name))[0]


def _load_benchmarks() -> dict:
    with BENCHMARKS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _signal_ai_score(value: float, signal: dict) -> float:
    direction = signal["direction"]

    if direction == "lower_is_ai":
        human = signal["human_center"]
        ai = signal["ai_center"]
        if value <= ai:
            return 1.0
        if value >= human:
            return 0.0
        return _clamp((human - value) / (human - ai))

    if direction == "higher_is_ai":
        human = signal["human_center"]
        ai = signal["ai_center"]
        if value >= ai:
            return 1.0
        if value <= human:
            return 0.0
        return _clamp((value - human) / (ai - human))

    if direction == "deviation_is_ai":
        center = signal["human_center"]
        spread = signal["spread"]
        return _clamp(abs(value - center) / spread)

    raise ValueError(f"Unknown direction: {direction}")


@lru_cache(maxsize=1)
def _load_nlp_md():
    import spacy

    return spacy.load("pl_core_news_md")


@lru_cache(maxsize=1)
def _basic_popularity_dict() -> dict[str, float]:
    if BASIC_DICT_CACHE.is_file():
        with BASIC_DICT_CACHE.open(encoding="utf-8") as handle:
            return json.load(handle)

    dataset_path = REPO_ROOT / "dataset" / "scraped_news.json"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Brak pliku dataset/scraped_news.json potrzebnego do bazowego indeksu popularności."
        )

    with dataset_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)

    texts = [row["body"] for row in rows if row.get("body")][:400]
    labels = [0] * len(texts)
    nlp = _load_nlp_md()
    popularity_dict = build_popularity_dictionary(texts, labels, nlp)

    with BASIC_DICT_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(popularity_dict, handle, ensure_ascii=False)

    return popularity_dict


def _resolve_wiki_dict_path() -> Path | None:
    for path in WIKI_DICT_CANDIDATES:
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=1)
def _wiki_popularity_dict() -> dict[str, float]:
    path = _resolve_wiki_dict_path()
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _popularity_index(text: str, popularity_dict: dict[str, float]) -> float:
    if not popularity_dict:
        return 0.0

    nlp = _load_nlp_md()
    doc = nlp(text)
    valid_words_count = 0
    total_score = 0.0

    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha:
            lemma = token.lemma_.lower()
            total_score += popularity_dict.get(lemma, 0.0)
            valid_words_count += 1

    if valid_words_count == 0:
        return 0.0
    return total_score / valid_words_count


def _adp_index() -> int:
    return list(UD_TAGS).index("ADP")


def extract_metric_values(text: str) -> dict[str, float]:
    cleaned = preprocess_for_bert(text)
    pos_ratios = all_pos_per_word(cleaned)

    values = {
        "avg_sentence_len": avg_sentence_len(cleaned),
        "ttr_lemmatized": ttr_lemmatized(cleaned),
        "pos_adp": pos_ratios["ADP"],
        "basic_popularity_index": _popularity_index(cleaned, _basic_popularity_dict()),
    }

    wiki_dict = _wiki_popularity_dict()
    if wiki_dict:
        values["wiki_popularity_index"] = _popularity_index(cleaned, wiki_dict)

    return values


def analyze_text(text: str, threshold: float = 0.5) -> AnalysisResult:
    stripped = text.strip()
    word_count = len(stripped.split())

    if not stripped:
        return AnalysisResult(
            label="—",
            probability_ai=0.0,
            probability_human=0.0,
            word_count=0,
            signals=[],
            reference_signals=[],
            warning="Wprowadź tekst do analizy.",
        )

    warning = None
    if word_count < MIN_WORDS:
        warning = (
            f"Tekst ma tylko {word_count} słów. Analiza EDA była wykonana na dłuższych "
            f"fragmentach (≥ {MIN_WORDS} słów) — wynik może być mniej wiarygodny."
        )

    benchmarks = _load_benchmarks()
    metric_values = extract_metric_values(stripped)

    signals: list[SignalResult] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for signal in benchmarks["signals"]:
        value = metric_values[signal["id"]]
        ai_score = _signal_ai_score(value, signal)
        weight = signal["weight"]
        weighted_sum += ai_score * weight
        weight_total += weight

        signals.append(
            SignalResult(
                signal_id=signal["id"],
                label=signal["label"],
                description=signal["description"],
                value=value,
                ai_score=ai_score,
                weight=weight,
                unit=signal.get("unit", ""),
                human_center=signal.get("human_center"),
                ai_center=signal.get("ai_center"),
                chart_path=_find_chart(
                    signal.get("chart_keyword"),
                    tuple(signal.get("chart_exclude", [])),
                ),
            )
        )

    probability_ai = weighted_sum / weight_total if weight_total else 0.0
    probability_human = 1.0 - probability_ai
    label = "Wygenerowany przez AI" if probability_ai >= threshold else "Prawdopodobnie ludzki"

    reference_signals: list[SignalResult] = []
    ref = benchmarks.get("reference_only")
    if ref and ref["id"] in metric_values:
        reference_signals.append(
            SignalResult(
                signal_id=ref["id"],
                label=ref["label"],
                description=ref["description"],
                value=metric_values[ref["id"]],
                ai_score=0.0,
                weight=0.0,
                unit="0–1",
            )
        )

    return AnalysisResult(
        label=label,
        probability_ai=probability_ai,
        probability_human=probability_human,
        word_count=word_count,
        signals=signals,
        reference_signals=reference_signals,
        warning=warning,
    )


def list_eda_charts() -> list[tuple[str, Path]]:
    benchmarks = _load_benchmarks()
    charts: list[tuple[str, Path]] = []

    for signal in benchmarks["signals"]:
        path = _find_chart(
            signal.get("chart_keyword"),
            tuple(signal.get("chart_exclude", [])),
        )
        if path is not None:
            charts.append((signal["label"], path))

    return charts
