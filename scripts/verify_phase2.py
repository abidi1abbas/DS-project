"""Run reproducible checks for the completed Phase 2 deliverables."""

import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"


def check_notebooks():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") is not None for cell in code_cells), path
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, path


def check_metrics():
    final_results = pd.read_csv(RESULTS / "model_results.csv").set_index("model")
    assert set(final_results.index) == {
        "Dummy Baseline",
        "Naive Bayes",
        "Logistic Regression",
        "Linear SVM",
        "BiLSTM",
        "BERT Tiny",
    }

    prediction_sources = [
        (RESULTS / "classical_test_predictions.csv", [
            "Dummy Baseline", "Naive Bayes", "Logistic Regression", "Linear SVM"
        ]),
        (RESULTS / "bilstm_test_predictions.csv", ["BiLSTM"]),
        (RESULTS / "bert_test_predictions.csv", ["BERT Tiny"]),
    ]

    for path, model_names in prediction_sources:
        predictions = pd.read_csv(path)
        assert len(predictions) == 3898, path
        for model_name in model_names:
            actual = predictions["label"]
            predicted = predictions[model_name]
            expected = final_results.loc[model_name]
            assert abs(accuracy_score(actual, predicted) - expected["accuracy"]) < 1e-12
            assert abs(f1_score(actual, predicted, average="macro") - expected["macro_f1"]) < 1e-12
            assert abs(f1_score(actual, predicted, average="weighted") - expected["weighted_f1"]) < 1e-12


def check_models():
    classical_model = joblib.load(MODELS / "best_classical_model.joblib")
    assert classical_model.predict(["yeh movie bohat achi hai"]).shape == (1,)

    bilstm_state = torch.load(MODELS / "bilstm_sentiment.pt", map_location="cpu", weights_only=True)
    bert_state = torch.load(MODELS / "bert_tiny_sentiment.pt", map_location="cpu", weights_only=True)
    assert bilstm_state and bert_state
    assert "output.weight" in bilstm_state
    assert "classifier.weight" in bert_state

    vocabulary = json.loads((MODELS / "bilstm_vocabulary.json").read_text(encoding="utf-8"))
    assert vocabulary["<pad>"] == 0 and vocabulary["<unk>"] == 1


def check_images_and_sizes():
    for number in range(6, 10):
        matches = list(IMAGES.glob(f"{number:02d}_*.png"))
        assert len(matches) == 1, f"Expected one image beginning with {number:02d}_"
        with Image.open(matches[0]) as image:
            image.verify()

    maximum_github_size = 100 * 1024 * 1024
    for folder in [MODELS, RESULTS, NOTEBOOKS, IMAGES]:
        for path in folder.rglob("*"):
            if path.is_file():
                assert path.stat().st_size < maximum_github_size, path


if __name__ == "__main__":
    check_notebooks()
    check_metrics()
    check_models()
    check_images_and_sizes()
    print("PASS: all Phase 2 notebooks, metrics, models, images, and file sizes are valid.")

