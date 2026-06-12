"""Predykcja człowiek vs AI — model BERT + cechy stylistyczne."""

from __future__ import annotations

from dataclasses import dataclass

from eda_scorer import AnalysisResult, SignalResult, analyze_text as analyze_eda_text
from model_predictor import ModelPrediction, predict_text as predict_with_model


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probability_ai: float
    probability_human: float
    word_count: int
    warning: str | None
    eda_signals: list[SignalResult]
    eda_reference_signals: list[SignalResult]


def predict(text: str, threshold: float = 0.5, include_eda_details: bool = True) -> PredictionResult:
    model_result: ModelPrediction = predict_with_model(text, threshold=threshold)

    eda_signals: list[SignalResult] = []
    eda_reference_signals: list[SignalResult] = []
    warning = model_result.warning

    if include_eda_details and model_result.word_count > 0:
        eda_result: AnalysisResult = analyze_eda_text(text, threshold=threshold)
        eda_signals = eda_result.signals
        eda_reference_signals = eda_result.reference_signals
        if warning is None:
            warning = eda_result.warning

    return PredictionResult(
        label=model_result.label,
        probability_ai=model_result.probability_ai,
        probability_human=model_result.probability_human,
        word_count=model_result.word_count,
        warning=warning,
        eda_signals=eda_signals,
        eda_reference_signals=eda_reference_signals,
    )


__all__ = ["PredictionResult", "predict", "predict_with_model"]
