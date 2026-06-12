from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

DM_DIR = Path(__file__).resolve().parent
REPO_ROOT = DM_DIR.parent
SRC_DIR = REPO_ROOT / "src"
MODEL_PATH = DM_DIR / "best_model.pt"
NKJP_DICT_PATH = REPO_ROOT / "nkjp_popularity_dict.json"
BERT_MODEL_NAME = "allegro/herbert-base-cased"
VECTOR_SIZE = 27
MAX_LENGTH = 512
MIN_WORDS = 15

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_news_detector.features.pos import UD_TAGS, all_pos_per_word
from ai_news_detector.features.punctuation import punctuation_per_letter, punctuation_per_word
from ai_news_detector.features.syllables import avg_syllables_per_sentence, avg_word_length
from ai_news_detector.features.text_stats import avg_sentence_len, capital_ratio, ttr, ttr_lemmatized
from feature_bert import MultiModalBertModel
from load_dataset import preprocess_for_bert


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    probability_ai: float
    probability_human: float
    word_count: int
    warning: str | None = None


@lru_cache(maxsize=1)
def _load_nlp_md():
    import spacy

    return spacy.load("pl_core_news_md")


@lru_cache(maxsize=1)
def _nkjp_popularity_dict() -> dict[str, float]:
    if not NKJP_DICT_PATH.is_file():
        raise FileNotFoundError(
            f"Brak pliku {NKJP_DICT_PATH.name} potrzebnego do indeksu popularności NKJP."
        )
    with NKJP_DICT_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _popularity_index(text: str, popularity_dict: dict[str, float]) -> float:
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


def extract_feature_vector(text: str) -> list[float]:
    cleaned = preprocess_for_bert(text)
    pos_ratios = all_pos_per_word(cleaned)

    features: list[float] = [pos_ratios[tag] for tag in UD_TAGS]
    features.extend(
        [
            punctuation_per_letter(cleaned),
            punctuation_per_word(cleaned),
            avg_sentence_len(cleaned),
            capital_ratio(cleaned),
            ttr(cleaned),
            ttr_lemmatized(cleaned),
            avg_syllables_per_sentence(cleaned),
            avg_word_length(cleaned),
            _popularity_index(cleaned, _nkjp_popularity_dict()),
        ]
    )

    if len(features) != VECTOR_SIZE:
        raise ValueError(f"Oczekiwano {VECTOR_SIZE} cech, otrzymano {len(features)}.")

    return features


def _normalize_features(
    features: list[float],
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
) -> torch.Tensor:
    vector = torch.tensor(features, dtype=torch.float32, device=feature_means.device)
    safe_stds = torch.where(feature_stds == 0, torch.ones_like(feature_stds), feature_stds)
    return (vector - feature_means) / safe_stds


@dataclass
class _LoadedModel:
    model: MultiModalBertModel
    tokenizer: AutoTokenizer
    device: torch.device
    feature_means: torch.Tensor
    feature_stds: torch.Tensor


@lru_cache(maxsize=1)
def _load_model_bundle() -> _LoadedModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Brak pliku modelu: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = MultiModalBertModel(
        bert_model_name=BERT_MODEL_NAME,
        vector_input_size=VECTOR_SIZE,
        num_classes=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    feature_means = checkpoint["feature_means"].to(device).float()
    feature_stds = checkpoint["feature_stds"].to(device).float()

    return _LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )


def predict_text(text: str, threshold: float = 0.5) -> ModelPrediction:
    stripped = text.strip()
    word_count = len(stripped.split())

    if not stripped:
        return ModelPrediction(
            label="—",
            probability_ai=0.0,
            probability_human=0.0,
            word_count=0,
            warning="Wprowadź tekst do analizy.",
        )

    warning = None
    if word_count < MIN_WORDS:
        warning = (
            f"Tekst ma tylko {word_count} słów. Model był trenowany na dłuższych "
            f"fragmentach (≥ {MIN_WORDS} słów) — wynik może być mniej wiarygodny."
        )

    bundle = _load_model_bundle()
    cleaned = preprocess_for_bert(stripped)
    raw_features = extract_feature_vector(stripped)
    float_vectors = _normalize_features(
        raw_features,
        bundle.feature_means,
        bundle.feature_stds,
    ).unsqueeze(0).to(bundle.device)

    encoding = bundle.tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(bundle.device)
    attention_mask = encoding["attention_mask"].to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(input_ids, attention_mask, float_vectors).view(-1)
        probability_ai = float(torch.sigmoid(logits).item())

    probability_human = 1.0 - probability_ai
    label = "Wygenerowany przez AI" if probability_ai >= threshold else "Prawdopodobnie ludzki"

    return ModelPrediction(
        label=label,
        probability_ai=probability_ai,
        probability_human=probability_human,
        word_count=word_count,
        warning=warning,
    )
