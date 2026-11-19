"""Build the beginner-friendly Phase 2 notebooks.

The notebooks are kept as the main deliverables. This helper only makes their
structure reproducible and is not needed for ordinary use.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(filename, title, cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [md(f"# {title}")] + cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(notebook, NOTEBOOKS / filename)


setup = code(
    """
from pathlib import Path
import os

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA = PROJECT_ROOT / "data" / "processed"
IMAGES = PROJECT_ROOT / "images"
MODELS = PROJECT_ROOT / "models"
RESULTS = PROJECT_ROOT / "results"

for folder in [IMAGES, MODELS, RESULTS]:
    folder.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
print("Project folder:", PROJECT_ROOT)
"""
)


classical_cells = [
    md(
        """
## Goal

Build understandable baseline models before neural networks. Every model uses the
same text and labels. We first compare models on a validation portion of the
training data, then refit them on all training rows and evaluate once on the
untouched test set.

The models are:

1. most-frequent dummy baseline;
2. Multinomial Naive Bayes with word TF-IDF;
3. Logistic Regression with word and character TF-IDF;
4. Linear SVM with word and character TF-IDF.

Character n-grams are useful for Roman Urdu because spelling is not standardized.
"""
    ),
    setup,
    code(
        """
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
"""
    ),
    md("## 1. Load the fixed Phase 1 train and test files"),
    code(
        """
train_df = pd.read_csv(DATA / "train.csv")
test_df = pd.read_csv(DATA / "test.csv")

X_train = train_df["model_text"].fillna("")
y_train = train_df["label"]
X_test = test_df["model_text"].fillna("")
y_test = test_df["label"]

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Test data will not be used during model selection.")
"""
    ),
    md(
        """
## 2. Create validation data

The validation set comes only from the training file. `stratify` preserves class
percentages, and the test set remains untouched until final evaluation.
"""
    ),
    code(
        """
X_fit, X_validation, y_fit, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train,
)

print("Fit rows:", len(X_fit))
print("Validation rows:", len(X_validation))
"""
    ),
    md("## 3. Define simple reusable pipelines"),
    code(
        """
word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=30000,
    sublinear_tf=True,
)

word_and_character_tfidf = FeatureUnion(
    [
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=40000,
                sublinear_tf=True,
            ),
        ),
    ]
)

models = {
    "Dummy Baseline": Pipeline(
        [("vectorizer", TfidfVectorizer(max_features=1000)),
         ("classifier", DummyClassifier(strategy="most_frequent"))]
    ),
    "Naive Bayes": Pipeline(
        [("vectorizer", word_tfidf),
         ("classifier", MultinomialNB(alpha=0.5))]
    ),
    "Logistic Regression": Pipeline(
        [("vectorizer", word_and_character_tfidf),
         ("classifier", LogisticRegression(
             C=2.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
         ))]
    ),
    "Linear SVM": Pipeline(
        [("vectorizer", word_and_character_tfidf),
         ("classifier", LinearSVC(C=1.5, class_weight="balanced", random_state=RANDOM_STATE))]
    ),
}

print("Models:", list(models))
"""
    ),
    md("## 4. Select a model using validation macro-F1"),
    code(
        """
validation_rows = []

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    predictions = model.predict(X_validation)
    seconds = time.perf_counter() - start

    validation_rows.append(
        {
            "model": name,
            "validation_accuracy": accuracy_score(y_validation, predictions),
            "validation_macro_f1": f1_score(y_validation, predictions, average="macro"),
            "validation_weighted_f1": f1_score(y_validation, predictions, average="weighted"),
            "validation_fit_seconds": seconds,
        }
    )
    print(f"Finished {name}: macro-F1 = {validation_rows[-1]['validation_macro_f1']:.4f}")

validation_results = pd.DataFrame(validation_rows).sort_values(
    "validation_macro_f1", ascending=False
).reset_index(drop=True)
validation_results.round(4)
"""
    ),
    md(
        """
Macro-F1 gives equal importance to Negative, Neutral, and Positive. We use it for
selection because the classes are not perfectly balanced.
"""
    ),
    md("## 5. Refit every model and evaluate on the untouched test set"),
    code(
        """
test_rows = []
prediction_table = test_df[["sentence", "sentiment", "label"]].copy()
confusion_matrices = {}

for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    seconds = time.perf_counter() - start

    test_rows.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "weighted_f1": f1_score(y_test, predictions, average="weighted"),
            "fit_seconds": seconds,
        }
    )
    prediction_table[name] = predictions
    confusion_matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    print(f"Finished {name}: test macro-F1 = {test_rows[-1]['macro_f1']:.4f}")

test_results = pd.DataFrame(test_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
test_results.round(4)
"""
    ),
    md("## 6. Detailed report and confusion matrices"),
    code(
        """
best_classical_name = validation_results.iloc[0]["model"]
best_predictions = prediction_table[best_classical_name]

print("Selected from validation data:", best_classical_name)
print(
    classification_report(
        y_test,
        best_predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
)
"""
    ),
    code(
        """
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for axis, (name, matrix) in zip(axes.ravel(), confusion_matrices.items()):
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=axis,
    )
    axis.set_title(name)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")

plt.suptitle("Classical Model Confusion Matrices", y=1.02, fontsize=15)
plt.tight_layout()
plt.savefig(IMAGES / "06_classical_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    md("## 7. Save the selected model and reproducible results"),
    code(
        """
selected_model = models[best_classical_name]

joblib.dump(selected_model, MODELS / "best_classical_model.joblib")
validation_results.to_csv(RESULTS / "classical_validation_results.csv", index=False)
test_results.to_csv(RESULTS / "classical_test_results.csv", index=False)
prediction_table.to_csv(RESULTS / "classical_test_predictions.csv", index=False)

(RESULTS / "best_classical_model.txt").write_text(best_classical_name, encoding="utf-8")

print("Saved selected model:", MODELS / "best_classical_model.joblib")
print("Selected model:", best_classical_name)
"""
    ),
    md(
        """
## Conclusion

The dummy result is the minimum benchmark. The selected TF-IDF classifier was
chosen using validation macro-F1, not the test result. The same fixed test set is
used later for comparison with the BiLSTM and transformer.
"""
    ),
]


bilstm_cells = [
    md(
        """
## Goal

Train a small bidirectional LSTM (BiLSTM) from scratch. Unlike TF-IDF, it reads the
word sequence in both directions. The vocabulary is built only from training data,
and validation data controls early stopping.
"""
    ),
    setup,
    code(
        """
import copy
import json
import random
import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
print("Device:", DEVICE)
"""
    ),
    md("## 1. Load data and make a training-only validation split"),
    code(
        """
train_df = pd.read_csv(DATA / "train.csv")
test_df = pd.read_csv(DATA / "test.csv")

fit_df, validation_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"],
)

fit_df = fit_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

print("Fit rows:", len(fit_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))
"""
    ),
    md("## 2. Tokenize text and build the vocabulary from fit data only"),
    code(
        """
def tokenize(text):
    # Keep URL/user markers and simple Roman Urdu word tokens.
    return re.findall(r"<url>|<user>|[a-zA-Z0-9']+", str(text).lower())


word_counts = Counter()
for text in fit_df["model_text"]:
    word_counts.update(tokenize(text))

MAX_VOCABULARY = 20000
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
word_to_id = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}

for word, _ in word_counts.most_common(MAX_VOCABULARY - 2):
    word_to_id[word] = len(word_to_id)

MAX_LENGTH = 50

print("Vocabulary size:", len(word_to_id))
print("Maximum sequence length:", MAX_LENGTH)
print("Example tokens:", tokenize(fit_df.loc[0, "model_text"])[:15])
"""
    ),
    md("## 3. Convert each sentence into a fixed-length sequence"),
    code(
        """
def encode_text(text):
    token_ids = [word_to_id.get(word, word_to_id[UNKNOWN_TOKEN]) for word in tokenize(text)]
    token_ids = token_ids[:MAX_LENGTH]
    token_ids += [word_to_id[PAD_TOKEN]] * (MAX_LENGTH - len(token_ids))
    return token_ids


class SentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["model_text"].tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.tensor(encode_text(self.texts[index]), dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


BATCH_SIZE = 128
fit_loader = DataLoader(SentimentDataset(fit_df), batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(SentimentDataset(validation_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(SentimentDataset(test_df), batch_size=BATCH_SIZE)

example_text, example_label = next(iter(fit_loader))
print("Batch text shape:", example_text.shape)
print("Batch label shape:", example_label.shape)
"""
    ),
    md("## 4. Define the compact BiLSTM"),
    code(
        """
class BiLSTMSentiment(nn.Module):
    def __init__(self, vocabulary_size, embedding_size=64, hidden_size=64, number_of_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.35)
        self.output = nn.Linear(hidden_size * 2, number_of_classes)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, (hidden, _) = self.lstm(embedded)
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        combined = torch.cat((forward_last, backward_last), dim=1)
        return self.output(self.dropout(combined))


model = BiLSTMSentiment(len(word_to_id)).to(DEVICE)
print(model)
print("Trainable parameters:", sum(parameter.numel() for parameter in model.parameters()))
"""
    ),
    md("## 5. Train with class weights and early stopping"),
    code(
        """
class_counts = fit_df["label"].value_counts().sort_index().values
class_weights = len(fit_df) / (3 * class_counts)
loss_function = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def evaluate(loader):
    model.eval()
    losses = []
    predictions = []
    actual = []

    with torch.no_grad():
        for token_ids, labels in loader:
            token_ids = token_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(token_ids)
            losses.append(loss_function(logits, labels).item())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.cpu().tolist())

    return np.mean(losses), f1_score(actual, predictions, average="macro"), predictions, actual


history = []
best_state = None
best_validation_f1 = -1
patience = 2
epochs_without_improvement = 0
training_start = time.perf_counter()

for epoch in range(1, 6):
    model.train()
    training_losses = []

    for token_ids, labels in fit_loader:
        token_ids = token_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(token_ids)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

    validation_loss, validation_f1, _, _ = evaluate(validation_loader)
    row = {
        "epoch": epoch,
        "train_loss": float(np.mean(training_losses)),
        "validation_loss": float(validation_loss),
        "validation_macro_f1": float(validation_f1),
    }
    history.append(row)
    print(row)

    if validation_f1 > best_validation_f1:
        best_validation_f1 = validation_f1
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping: validation macro-F1 stopped improving.")
            break

training_seconds = time.perf_counter() - training_start
model.load_state_dict(best_state)
print(f"Training time: {training_seconds:.1f} seconds")
"""
    ),
    md("## 6. Evaluate once on the untouched test set"),
    code(
        """
test_loss, test_macro_f1, test_predictions, test_actual = evaluate(test_loader)
test_accuracy = accuracy_score(test_actual, test_predictions)
test_weighted_f1 = f1_score(test_actual, test_predictions, average="weighted")

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test macro-F1: {test_macro_f1:.4f}")
print(
    classification_report(
        test_actual,
        test_predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
)
"""
    ),
    code(
        """
history_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train")
axes[0].plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="Validation")
axes[0].set_title("BiLSTM Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

matrix = confusion_matrix(test_actual, test_predictions, labels=[0, 1, 2])
sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
axes[1].set_title("BiLSTM Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(IMAGES / "07_bilstm_training_and_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    md("## 7. Save the model, vocabulary, predictions, and metrics"),
    code(
        """
torch.save(model.state_dict(), MODELS / "bilstm_sentiment.pt")
(MODELS / "bilstm_vocabulary.json").write_text(
    json.dumps(word_to_id, ensure_ascii=False), encoding="utf-8"
)

configuration = {
    "max_length": MAX_LENGTH,
    "embedding_size": 64,
    "hidden_size": 64,
    "labels": {"0": "Negative", "1": "Neutral", "2": "Positive"},
}
(MODELS / "bilstm_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")

pd.DataFrame(history).to_csv(RESULTS / "bilstm_training_history.csv", index=False)
pd.DataFrame(
    [{
        "model": "BiLSTM",
        "accuracy": test_accuracy,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
        "fit_seconds": training_seconds,
    }]
).to_csv(RESULTS / "bilstm_test_results.csv", index=False)

bilstm_predictions = test_df[["sentence", "sentiment", "label"]].copy()
bilstm_predictions["BiLSTM"] = test_predictions
bilstm_predictions.to_csv(RESULTS / "bilstm_test_predictions.csv", index=False)

print("Saved BiLSTM artifacts and results.")
"""
    ),
    md(
        """
## Conclusion

The BiLSTM learns sequential information without pretrained language knowledge.
Early stopping keeps the best validation checkpoint and reduces overfitting.
"""
    ),
]


transformer_cells = [
    md(
        """
## Goal

Fine-tune a compact pretrained BERT model for three-class Roman Urdu sentiment.
`google/bert_uncased_L-2_H-128_A-2` is a small two-layer BERT checkpoint chosen so
the experiment is reproducible on a CPU-only laptop. Its English WordPiece
vocabulary can split Latin-script Roman Urdu, but this limitation is discussed in
the final comparison.
"""
    ),
    setup,
    code(
        """
import copy
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
print("Device:", DEVICE)
print("Checkpoint:", MODEL_NAME)
"""
    ),
    md("## 1. Load data and create validation data from training rows"),
    code(
        """
train_df = pd.read_csv(DATA / "train.csv")
test_df = pd.read_csv(DATA / "test.csv")

fit_df, validation_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"],
)
fit_df = fit_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)

print("Fit rows:", len(fit_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))
"""
    ),
    md("## 2. Download the tokenizer and pretrained BERT weights"),
    code(
        """
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
    label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
).to(DEVICE)

print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))
print("Example tokens:", tokenizer.tokenize("yeh film bohat achi hai"))
"""
    ),
    md("## 3. Tokenize sentences into BERT inputs"),
    code(
        """
MAX_LENGTH = 64


class TransformerDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["model_text"].fillna("").tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = tokenizer(
            self.texts[index],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


BATCH_SIZE = 64
fit_loader = DataLoader(TransformerDataset(fit_df), batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(TransformerDataset(validation_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(TransformerDataset(test_df), batch_size=BATCH_SIZE)

example = next(iter(fit_loader))
print({key: value.shape for key, value in example.items()})
"""
    ),
    md("## 4. Fine-tune BERT with class-weighted loss"),
    code(
        """
class_counts = fit_df["label"].value_counts().sort_index().values
class_weights = len(fit_df) / (3 * class_counts)
loss_function = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)


def forward_batch(batch):
    labels = batch.pop("labels").to(DEVICE)
    inputs = {key: value.to(DEVICE) for key, value in batch.items()}
    logits = model(**inputs).logits
    return logits, labels


def evaluate(loader):
    model.eval()
    losses = []
    predictions = []
    actual = []

    with torch.no_grad():
        for batch in loader:
            logits, labels = forward_batch(batch)
            losses.append(loss_function(logits, labels).item())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.cpu().tolist())

    return np.mean(losses), f1_score(actual, predictions, average="macro"), predictions, actual


history = []
best_state = None
best_validation_f1 = -1
training_start = time.perf_counter()

for epoch in range(1, 3):
    model.train()
    training_losses = []

    for batch in fit_loader:
        optimizer.zero_grad()
        logits, labels = forward_batch(batch)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

    validation_loss, validation_f1, _, _ = evaluate(validation_loader)
    row = {
        "epoch": epoch,
        "train_loss": float(np.mean(training_losses)),
        "validation_loss": float(validation_loss),
        "validation_macro_f1": float(validation_f1),
    }
    history.append(row)
    print(row)

    if validation_f1 > best_validation_f1:
        best_validation_f1 = validation_f1
        best_state = copy.deepcopy(model.state_dict())

training_seconds = time.perf_counter() - training_start
model.load_state_dict(best_state)
print(f"Training time: {training_seconds:.1f} seconds")
"""
    ),
    md("## 5. Evaluate once on the same untouched test set"),
    code(
        """
test_loss, test_macro_f1, test_predictions, test_actual = evaluate(test_loader)
test_accuracy = accuracy_score(test_actual, test_predictions)
test_weighted_f1 = f1_score(test_actual, test_predictions, average="weighted")

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test macro-F1: {test_macro_f1:.4f}")
print(
    classification_report(
        test_actual,
        test_predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
)
"""
    ),
    code(
        """
history_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train")
axes[0].plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="Validation")
axes[0].set_title("BERT Fine-tuning Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

matrix = confusion_matrix(test_actual, test_predictions, labels=[0, 1, 2])
sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
axes[1].set_title("BERT Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(IMAGES / "08_bert_training_and_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    md("## 6. Save model state, settings, predictions, and metrics"),
    code(
        """
torch.save(model.state_dict(), MODELS / "bert_tiny_sentiment.pt")

configuration = {
    "base_model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "labels": {"0": "Negative", "1": "Neutral", "2": "Positive"},
}
(MODELS / "bert_tiny_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")

pd.DataFrame(history).to_csv(RESULTS / "bert_training_history.csv", index=False)
pd.DataFrame(
    [{
        "model": "BERT Tiny",
        "accuracy": test_accuracy,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
        "fit_seconds": training_seconds,
    }]
).to_csv(RESULTS / "bert_test_results.csv", index=False)

bert_predictions = test_df[["sentence", "sentiment", "label"]].copy()
bert_predictions["BERT Tiny"] = test_predictions
bert_predictions.to_csv(RESULTS / "bert_test_predictions.csv", index=False)

print("Saved BERT artifacts and results.")
"""
    ),
    md(
        """
## Conclusion

This is a genuine fine-tuning experiment: pretrained BERT weights are updated for
the sentiment labels. The checkpoint is intentionally tiny for CPU reproducibility;
a larger multilingual transformer is a sensible future improvement when a GPU is
available.
"""
    ),
]


comparison_cells = [
    md(
        """
## Goal

Combine every final test result, compare models fairly, inspect errors, and provide
a reusable prediction example. All metrics below come from the same untouched
3,898-row test set.
"""
    ),
    setup,
    code(
        """
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
"""
    ),
    md("## 1. Combine final model results"),
    code(
        """
classical = pd.read_csv(RESULTS / "classical_test_results.csv")
bilstm = pd.read_csv(RESULTS / "bilstm_test_results.csv")
bert = pd.read_csv(RESULTS / "bert_test_results.csv")

all_results = pd.concat([classical, bilstm, bert], ignore_index=True)
all_results = all_results.sort_values("macro_f1", ascending=False).reset_index(drop=True)
all_results.to_csv(RESULTS / "model_results.csv", index=False)

all_results.round(4)
"""
    ),
    code(
        """
plot_data = all_results.melt(
    id_vars="model",
    value_vars=["accuracy", "macro_f1", "weighted_f1"],
    var_name="metric",
    value_name="score",
)

plt.figure(figsize=(11, 6))
ax = sns.barplot(data=plot_data, x="model", y="score", hue="metric")
plt.ylim(0, 1)
plt.title("Final Model Comparison on the Same Test Set")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(IMAGES / "09_final_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    md("## 2. Select the best model by macro-F1"),
    code(
        """
best_model_name = all_results.loc[0, "model"]
best_score = all_results.loc[0, "macro_f1"]

print("Best final model:", best_model_name)
print(f"Test macro-F1: {best_score:.4f}")
print("Macro-F1 is the main comparison metric because it treats every class equally.")
"""
    ),
    md("## 3. Error analysis"),
    code(
        """
if best_model_name in ["Dummy Baseline", "Naive Bayes", "Logistic Regression", "Linear SVM"]:
    predictions = pd.read_csv(RESULTS / "classical_test_predictions.csv")
    predictions["predicted_label"] = predictions[best_model_name]
elif best_model_name == "BiLSTM":
    predictions = pd.read_csv(RESULTS / "bilstm_test_predictions.csv")
    predictions["predicted_label"] = predictions["BiLSTM"]
else:
    predictions = pd.read_csv(RESULTS / "bert_test_predictions.csv")
    predictions["predicted_label"] = predictions["BERT Tiny"]

predictions["predicted_sentiment"] = predictions["predicted_label"].map(LABEL_MAP)
errors = predictions[predictions["label"] != predictions["predicted_label"]].copy()
errors["word_count"] = errors["sentence"].fillna("").str.split().str.len()


def likely_difficulty(row):
    text = str(row["sentence"]).lower()
    if row["word_count"] <= 3:
        return "very short or ambiguous"
    if any(word in text.split() for word in ["lekin", "lakin", "magar", "but"]):
        return "mixed or contrasting opinion"
    if any(word in text.split() for word in ["movie", "film", "the", "is", "good", "bad"]):
        return "Urdu-English code-switching"
    if "!" in text or "?" in text:
        return "informal expression or punctuation"
    return "context, spelling variation, or possible label ambiguity"


errors["likely_difficulty"] = errors.apply(likely_difficulty, axis=1)
errors.to_csv(RESULTS / "all_best_model_errors.csv", index=False)

sample_groups = []
for _, group in errors.groupby("likely_difficulty"):
    sample_groups.append(group.sample(min(10, len(group)), random_state=42))

error_sample = pd.concat(sample_groups, ignore_index=True)
error_sample.to_csv(RESULTS / "error_analysis_sample.csv", index=False)

print("Total test errors:", len(errors))
errors["likely_difficulty"].value_counts().to_frame("count")
"""
    ),
    code(
        """
pd.set_option("display.max_colwidth", 120)
error_sample[[
    "sentence", "sentiment", "predicted_sentiment", "likely_difficulty"
]].head(15)
"""
    ),
    md(
        """
The difficulty categories are transparent heuristics for organizing manual review;
they are not new ground-truth labels. They highlight short ambiguous text,
contrast words, code-switching, punctuation, and remaining context/spelling issues.
"""
    ),
    md("## 4. Beginner-friendly prediction example using the deployable classical model"),
    code(
        """
classical_model = joblib.load(MODELS / "best_classical_model.joblib")


def predict_sentiment(text):
    numeric_prediction = int(classical_model.predict([text])[0])
    return LABEL_MAP[numeric_prediction]


examples = [
    "yeh movie bohat achi hai",
    "mujhe bilkul pasand nahi aya",
    "film kal release hui thi",
]

for example in examples:
    print(f"{example!r} -> {predict_sentiment(example)}")
"""
    ),
    md("## 5. Final findings"),
    code(
        """
best_row = all_results.iloc[0]
dummy_row = all_results[all_results["model"] == "Dummy Baseline"].iloc[0]
improvement = best_row["macro_f1"] - dummy_row["macro_f1"]

print(f"1. Best model: {best_row['model']}.")
print(f"2. Best accuracy: {best_row['accuracy']:.4f}.")
print(f"3. Best macro-F1: {best_row['macro_f1']:.4f}.")
print(f"4. Macro-F1 improvement over dummy baseline: {improvement:.4f}.")
print(f"5. Number of errors reviewed and exported: {len(error_sample)}.")
print("6. Roman Urdu spelling variation and short ambiguous messages remain important challenges.")
"""
    ),
    md(
        """
## Conclusion

Phase 2 now contains classical, deep-learning, and transformer experiments with a
shared test set, class-aware metrics, confusion matrices, model comparison, saved
artifacts, error analysis, and an inference example. An LLM prompting experiment
was left as an optional future extension because it would require an external API
and would not be directly reproducible from this repository.
"""
    ),
]


save("04_classical_models.ipynb", "Phase 2 — Classical Models", classical_cells)
save("05_bilstm.ipynb", "Phase 2 — BiLSTM", bilstm_cells)
save("06_transformer.ipynb", "Phase 2 — Transformer Fine-tuning", transformer_cells)
save("07_model_comparison_and_error_analysis.ipynb", "Phase 2 — Model Comparison and Error Analysis", comparison_cells)

print("Created Phase 2 notebooks in", NOTEBOOKS)
