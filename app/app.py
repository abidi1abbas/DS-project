"""Streamlit website for the Roman Urdu sentiment classifier."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_classical_model.joblib"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_results.csv"
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


@st.cache_resource
def load_model():
    """Load the trained TF-IDF Logistic Regression pipeline once."""
    return joblib.load(MODEL_PATH)


def predict(text):
    """Return the sentiment name and probability table for one sentence."""
    model = load_model()
    label = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]

    probability_table = pd.DataFrame(
        {
            "Sentiment": [LABEL_NAMES[int(number)] for number in model.classes_],
            "Confidence": probabilities,
        }
    ).sort_values("Confidence", ascending=False)

    return LABEL_NAMES[label], probability_table


st.set_page_config(
    page_title="Roman Urdu Sentiment Analysis",
    page_icon="💬",
    layout="centered",
)

st.title("Roman Urdu Sentiment Analysis")
st.write(
    "Enter a Roman Urdu sentence and the trained model will classify it as "
    "Positive, Negative, or Neutral."
)

sentence = st.text_area(
    "Roman Urdu sentence",
    value="yeh movie bohat achi hai",
    height=120,
    help="Example: mujhe yeh product bilkul pasand nahi aya",
)

if st.button("Analyze sentiment", type="primary", width="stretch"):
    if not sentence.strip():
        st.warning("Please enter a sentence before clicking Analyze sentiment.")
    else:
        sentiment, confidence = predict(sentence.strip())
        st.success(f"Predicted sentiment: **{sentiment}**")
        st.subheader("Model confidence")
        st.dataframe(
            confidence.style.format({"Confidence": "{:.2%}"}),
            hide_index=True,
            width="stretch",
        )
        st.bar_chart(confidence.set_index("Sentiment"))

st.divider()
st.subheader("Project results")
results = pd.read_csv(RESULTS_PATH)[["model", "accuracy", "macro_f1"]].copy()
results.columns = ["Model", "Accuracy", "Macro-F1"]
st.dataframe(
    results.style.format({"Accuracy": "{:.4f}", "Macro-F1": "{:.4f}"}),
    hide_index=True,
    width="stretch",
)

with st.expander("How this website works and its limitations"):
    st.markdown(
        """
        The deployed classifier is Logistic Regression with word and character
        TF-IDF features. It achieved **68.11% accuracy** and **0.6767 macro-F1** on
        the untouched test set.

        Roman Urdu has no standard spelling, and very short, sarcastic, mixed, or
        code-switched messages can be ambiguous. The confidence values are model
        estimates and should not be treated as certainty.
        """
    )

