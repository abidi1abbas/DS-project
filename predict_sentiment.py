"""Predict the sentiment of one Roman Urdu sentence.

Example:
    python predict_sentiment.py "yeh movie bohat achi hai"
"""

import sys
from pathlib import Path

import joblib


LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
MODEL_PATH = Path(__file__).parent / "models" / "best_classical_model.joblib"


def predict_sentiment(text):
    model = joblib.load(MODEL_PATH)
    predicted_label = int(model.predict([text])[0])
    return LABEL_NAMES[predicted_label]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict_sentiment.py "your Roman Urdu sentence"')
        raise SystemExit(1)

    sentence = " ".join(sys.argv[1:])
    print("Sentence:", sentence)
    print("Predicted sentiment:", predict_sentiment(sentence))
