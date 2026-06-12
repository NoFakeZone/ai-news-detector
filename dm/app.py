import sys
from pathlib import Path

import streamlit as st

DM_DIR = Path(__file__).resolve().parent
if str(DM_DIR) not in sys.path:
    sys.path.insert(0, str(DM_DIR))

from config import AI_THRESHOLD, BERT_MODEL_NAME, MODEL_PATH
from eda_scorer import SignalResult
from predictor import predict


st.set_page_config(
    page_title="Detektor wiadomości AI",
    page_icon="📰",
    layout="wide",
)

st.title("Detektor wiadomości AI")
st.caption("Klasyfikacja tekstu — człowiek vs AI (model Herbert + cechy stylistyczne)")


@st.cache_resource(show_spinner="Ładowanie modelu BERT...")
def _warm_up_model():
    from model_predictor import _load_model_bundle

    _load_model_bundle()
    return True


def _render_signal(signal: SignalResult) -> None:
    st.markdown(f"**{signal.label}** — {signal.description}")
    cols = st.columns([2, 3, 1])
    cols[0].metric("Wartość tekstu", f"{signal.value:.4f}")
    cols[1].progress(
        signal.ai_score,
        text=f"Wkład w ocenę AI: {signal.ai_score * 100:.0f}% (waga {signal.weight:.0%})",
    )
    if signal.human_center is not None and signal.ai_center is not None:
        cols[2].caption(f"Ludzie ≈ {signal.human_center:.3g}")
        cols[2].caption(f"AI ≈ {signal.ai_center:.3g}")
    elif signal.human_center is not None:
        cols[2].caption(f"Ludzie ≈ {signal.human_center:.3g}")


def _render_result(result, threshold: float) -> None:
    st.subheader("Wynik")

    ai_percent = result.probability_ai * 100
    human_percent = result.probability_human * 100

    if result.probability_ai >= threshold:
        st.error(f"**{result.label}**")
    else:
        st.success(f"**{result.label}**")

    st.progress(result.probability_ai, text=f"Prawdopodobieństwo AI: {ai_percent:.1f}%")

    metric_left, metric_right, metric_words = st.columns(3)
    metric_left.metric("AI", f"{ai_percent:.1f}%")
    metric_right.metric("Człowiek", f"{human_percent:.1f}%")
    metric_words.metric("Słowa", result.word_count)

    if result.warning:
        st.warning(result.warning)

    if result.eda_signals:
        st.markdown("### Sygnały pomocnicze z analizy EDA")
        st.caption("Poniższe wykresy mają charakter wyjaśniający — decyzję podejmuje wytrenowany model.")
        for signal in result.eda_signals:
            with st.container(border=True):
                _render_signal(signal)
                if signal.chart_path is not None:
                    with st.expander("Wykres referencyjny z EDA"):
                        st.image(str(signal.chart_path), use_container_width=True)

    if result.eda_reference_signals:
        st.markdown("### Informacyjnie (bez wpływu na wynik modelu)")
        for signal in result.eda_reference_signals:
            with st.container(border=True):
                st.markdown(f"**{signal.label}** — {signal.description}")
                st.metric("Wartość tekstu", f"{signal.value:.4f}")


with st.sidebar:
    st.header("Ustawienia")
    threshold = st.slider(
        "Próg klasyfikacji AI",
        min_value=0.0,
        max_value=1.0,
        value=AI_THRESHOLD,
        step=0.05,
    )
    st.markdown(
        f"**Model:** `{BERT_MODEL_NAME}`\n\n"
        f"**Plik wag:** `{MODEL_PATH.name}`\n\n"
        "Klasyfikator łączy reprezentację BERT (Herbert) z 27 cechami "
        "stylistycznymi (POS, interpunkcja, statystyki tekstu, indeks NKJP)."
    )

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.text_area(
    "Wprowadź tekst wiadomości",
    height=220,
    placeholder="Wklej lub wpisz fragment artykułu...",
    key="input_text",
)

col_analyze, col_clear = st.columns([1, 1])

with col_analyze:
    analyze_clicked = st.button("Analizuj", type="primary", use_container_width=True)

with col_clear:
    if st.button("Wyczyść", use_container_width=True):
        st.session_state.input_text = ""
        st.rerun()

if analyze_clicked:
    if not st.session_state.input_text.strip():
        st.warning("Wprowadź tekst przed analizą.")
    else:
        try:
            with st.spinner("Ładowanie modelu i analiza tekstu..."):
                _warm_up_model()
                result = predict(st.session_state.input_text, threshold=threshold)
            _render_result(result, threshold)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Wystąpił błąd podczas analizy: {exc}")

st.divider()
st.markdown(
    "**Uwaga:** Model był trenowany na polskich wiadomościach i tekstach generowanych przez LLM. "
    "Wynik ma charakter probabilistyczny i nie zastępuje weryfikacji źródeł."
)
