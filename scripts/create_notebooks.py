"""Create the three beginner-friendly Phase 1 notebooks.

Run this file only when the notebook templates need to be rebuilt. The notebooks
themselves contain all analysis code and can be opened and run normally.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save_notebook(filename, title, cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [markdown(f"# {title}")] + cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(notebook, NOTEBOOKS / filename)


common_setup = code(
    """
from pathlib import Path
import os

# This makes the notebook work whether it is opened from the project folder
# or from inside the notebooks folder.
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
IMAGES = PROJECT_ROOT / "images"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

# Store the downloaded Hugging Face files inside this project.
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")

print("Project folder:", PROJECT_ROOT)
"""
)


cleaning_cells = [
    markdown(
        """
## Purpose

This notebook downloads the Roman Urdu sentiment dataset, checks its quality,
and creates a cleaned CSV file. The cleaning is deliberately conservative:
Roman Urdu has many valid spellings, so we do **not** try to correct spelling or
remove common words.
"""
    ),
    common_setup,
    code(
        """
import re
import unicodedata

import pandas as pd
from datasets import load_dataset

pd.set_option("display.max_colwidth", 120)
"""
    ),
    markdown(
        """
## 1. Load and inspect the dataset

`load_dataset` downloads the data the first time. Later runs use the local cache.
The dataset contains one train split, so we will create our own train/test split
in the preprocessing notebook.
"""
    ),
    code(
        """
dataset = load_dataset("community-datasets/roman_urdu")
dataset
"""
    ),
    code(
        """
train_data = dataset["train"]
print("Number of rows:", len(train_data))
print("Columns:", train_data.column_names)
print("Features:", train_data.features)
train_data[:5]
"""
    ),
    markdown(
        """
The sentiment is stored as a number internally. We use the dataset's own label
names to convert it into readable text instead of guessing what each number means.
"""
    ),
    code(
        """
df = train_data.to_pandas()
label_names = train_data.features["sentiment"].names
df["sentiment"] = df["sentiment"].map(
    {number: name for number, name in enumerate(label_names)}
)

df.head()
"""
    ),
    code(
        """
print("Shape:", df.shape)
print("\\nData types:")
print(df.dtypes)
print("\\nMissing values:")
print(df.isna().sum())
print("\\nClass counts:")
print(df["sentiment"].value_counts())
"""
    ),
    markdown(
        """
## 2. Standardize text format

The function below makes formatting consistent without changing the meaning:

- normalize Unicode characters;
- remove invisible characters;
- strip spaces at both ends;
- replace repeated spaces/tabs/newlines with one space;
- create a lowercase version for duplicate checking and later modeling.

We keep the readable original sentence in `sentence`.
"""
    ),
    code(
        """
def clean_sentence(text):
    # Apply small, safe formatting fixes to one sentence.
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\\u200b-\\u200d\\ufeff]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


df["sentence"] = df["sentence"].apply(clean_sentence)
df["clean_text"] = df["sentence"].str.lower()

df[["sentence", "clean_text", "sentiment"]].head()
"""
    ),
    markdown(
        """
## 3. Handle missing, empty, duplicate, and conflicting rows

Missing/empty text cannot teach a sentiment model anything, so it is removed.
Exact repeated text with the same label is also removed. If the same normalized
sentence has different labels, all copies are removed because there is no safe way
to decide which label is correct without manual relabeling.
"""
    ),
    code(
        """
starting_rows = len(df)

missing_label_count = df["sentiment"].isna().sum()
empty_text_count = df["clean_text"].eq("").sum()

df = df.dropna(subset=["sentiment"])
df = df[df["clean_text"] != ""].copy()

# A conflict means identical normalized text appears with more than one sentiment.
labels_per_text = df.groupby("clean_text")["sentiment"].nunique()
conflicting_texts = labels_per_text[labels_per_text > 1].index
conflicting_row_count = df["clean_text"].isin(conflicting_texts).sum()

df = df[~df["clean_text"].isin(conflicting_texts)].copy()

duplicate_count = df.duplicated(subset=["clean_text", "sentiment"]).sum()
df = df.drop_duplicates(subset=["clean_text", "sentiment"]).reset_index(drop=True)

print("Starting rows:", starting_rows)
print("Rows with missing sentiment:", missing_label_count)
print("Rows with empty text:", empty_text_count)
print("Rows removed because of conflicting labels:", conflicting_row_count)
print("Duplicate rows removed:", duplicate_count)
print("Final rows:", len(df))
"""
    ),
    markdown(
        """
## 4. Check text-length outliers

For numeric data, outliers are often removed. In text data, a very short or long
sentence can still be valid. We therefore inspect length outliers but keep them.
Only empty text has been removed.
"""
    ),
    code(
        """
df["character_count"] = df["clean_text"].str.len()
df["word_count"] = df["clean_text"].str.split().str.len()

q1 = df["word_count"].quantile(0.25)
q3 = df["word_count"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr
length_outliers = df[df["word_count"] > upper_limit]

print(df[["character_count", "word_count"]].describe().round(2))
print("\\nIQR upper limit for word count:", upper_limit)
print("Long-text outliers kept:", len(length_outliers))

length_outliers.nlargest(5, "word_count")[["sentence", "sentiment", "word_count"]]
"""
    ),
    markdown(
        """
## 5. Final quality checks and save files

The raw CSV is useful for reproducibility. The cleaned CSV is the input to the EDA
and preprocessing notebooks. Length columns are exploratory features, not changes
to the original target.
"""
    ),
    code(
        """
valid_labels = {"Positive", "Negative", "Neutral"}

print("Missing values after cleaning:", int(df.isna().sum().sum()))
print("Remaining duplicate normalized texts:", int(df["clean_text"].duplicated().sum()))
print("Labels are valid:", set(df["sentiment"]).issubset(valid_labels))
print("Final shape:", df.shape)
print("\\nFinal class counts:")
print(df["sentiment"].value_counts())
"""
    ),
    code(
        """
# Save the original converted data and the cleaned data.
raw_df = train_data.to_pandas()
raw_df["sentiment"] = raw_df["sentiment"].map(
    {number: name for number, name in enumerate(label_names)}
)

raw_path = DATA_RAW / "roman_urdu_raw.csv"
clean_path = DATA_PROCESSED / "roman_urdu_cleaned.csv"
summary_path = DATA_PROCESSED / "cleaning_summary.csv"

raw_df.to_csv(raw_path, index=False, encoding="utf-8")
df.to_csv(clean_path, index=False, encoding="utf-8")

cleaning_summary = pd.DataFrame(
    {
        "check": [
            "starting_rows",
            "missing_sentiment_rows",
            "empty_text_rows",
            "conflicting_label_rows",
            "duplicate_rows",
            "final_rows",
        ],
        "count": [
            starting_rows,
            missing_label_count,
            empty_text_count,
            conflicting_row_count,
            duplicate_count,
            len(df),
        ],
    }
)
cleaning_summary.to_csv(summary_path, index=False)

print("Saved:", raw_path)
print("Saved:", clean_path)
print("Saved:", summary_path)
cleaning_summary
"""
    ),
    markdown(
        """
## Cleaning conclusion

The resulting file has no missing labels, empty sentences, repeated normalized
sentences, or known label conflicts. Long messages remain because length alone
does not make a text record invalid. This avoids deleting useful language examples.
"""
    ),
]


eda_cells = [
    markdown(
        """
## Purpose

This notebook explores the cleaned Roman Urdu data with simple tables and charts.
It focuses on class balance, sentence lengths, common words, and relationships
between numeric text-length features.
"""
    ),
    common_setup,
    code(
        """
from collections import Counter
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 100)

df = pd.read_csv(DATA_PROCESSED / "roman_urdu_cleaned.csv")
print("Cleaned data shape:", df.shape)
df.head()
"""
    ),
    markdown("## 1. Class distribution"),
    code(
        """
class_counts = df["sentiment"].value_counts()
class_percentages = (df["sentiment"].value_counts(normalize=True) * 100).round(2)

class_summary = pd.DataFrame(
    {"count": class_counts, "percentage": class_percentages}
)
class_summary
"""
    ),
    code(
        """
class_order = ["Positive", "Negative", "Neutral"]
colors = ["#2a9d8f", "#e76f51", "#e9c46a"]

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="sentiment", order=class_order, palette=colors, hue="sentiment", legend=False)
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Sentences")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.savefig(IMAGES / "01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
The chart checks class imbalance. Accuracy alone may be misleading if one class is
much larger, so Phase 2 should also report macro precision, recall, and F1-score.
"""
    ),
    markdown("## 2. Sentence-length distributions"),
    code(
        """
# Recalculate to show exactly how these simple features are made.
df["character_count"] = df["clean_text"].str.len()
df["word_count"] = df["clean_text"].str.split().str.len()
df["average_word_length"] = (
    df["character_count"] / df["word_count"].replace(0, 1)
)
df["punctuation_count"] = df["sentence"].str.count(r"[^\\w\\s]")

df.groupby("sentiment")[["word_count", "character_count"]].agg(
    ["mean", "median", "max"]
).round(2)
"""
    ),
    code(
        """
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="word_count", hue="sentiment", bins=40, element="step", stat="density", common_norm=False)
plt.xlim(0, df["word_count"].quantile(0.99))
plt.title("Word Count Distribution by Sentiment (up to 99th percentile)")
plt.xlabel("Words per Sentence")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(IMAGES / "02_word_count_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="sentiment", y="word_count", order=class_order, palette=colors, hue="sentiment", legend=False)
plt.ylim(0, df["word_count"].quantile(0.99))
plt.title("Word Count by Sentiment (up to 99th percentile)")
plt.xlabel("Sentiment")
plt.ylabel("Words per Sentence")
plt.tight_layout()
plt.savefig(IMAGES / "03_word_count_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("## 3. Most common words"),
    code(
        """
def get_words(text_series):
    # Return simple alphabetic Roman Urdu tokens from a group of sentences.
    all_text = " ".join(text_series.astype(str))
    return re.findall(r"[a-zA-Z]+", all_text.lower())


top_words_by_sentiment = {}
for sentiment in class_order:
    words = get_words(df.loc[df["sentiment"] == sentiment, "clean_text"])
    top_words_by_sentiment[sentiment] = Counter(words).most_common(15)

pd.DataFrame(
    {
        sentiment: [f"{word} ({count})" for word, count in values]
        for sentiment, values in top_words_by_sentiment.items()
    }
)
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for axis, sentiment, color in zip(axes, class_order, colors):
    word_counts = top_words_by_sentiment[sentiment][::-1]
    words = [item[0] for item in word_counts]
    counts = [item[1] for item in word_counts]
    axis.barh(words, counts, color=color)
    axis.set_title(f"Top Words: {sentiment}")
    axis.set_xlabel("Count")

plt.tight_layout()
plt.savefig(IMAGES / "04_top_words_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
Common words are retained here because a reliable Roman Urdu stop-word list is not
available in the dataset. In Phase 2, TF-IDF can reduce the influence of words that
appear everywhere, while n-grams can preserve short sentiment phrases.
"""
    ),
    markdown("## 4. Correlation heatmap for derived numeric features"),
    code(
        """
numeric_columns = [
    "character_count",
    "word_count",
    "average_word_length",
    "punctuation_count",
]
correlations = df[numeric_columns].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(correlations, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Between Text-Length Features")
plt.tight_layout()
plt.savefig(IMAGES / "05_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

correlations
"""
    ),
    markdown("## 5. Automatically calculated key findings"),
    code(
        """
largest_class = class_counts.idxmax()
smallest_class = class_counts.idxmin()
imbalance_ratio = class_counts.max() / class_counts.min()
median_words = df["word_count"].median()
short_share = (df["word_count"] <= 5).mean() * 100
long_share = (df["word_count"] > 20).mean() * 100

print(f"1. The largest class is {largest_class} ({class_counts[largest_class]:,} rows).")
print(f"2. The smallest class is {smallest_class} ({class_counts[smallest_class]:,} rows).")
print(f"3. Largest-to-smallest class ratio: {imbalance_ratio:.2f}:1.")
print(f"4. Median sentence length: {median_words:.0f} words.")
print(f"5. Sentences with 5 words or fewer: {short_share:.1f}%.")
print(f"6. Sentences longer than 20 words: {long_share:.1f}%.")
print("7. Character count and word count are expected to be strongly related; both should not automatically be treated as independent signals.")
"""
    ),
    markdown(
        """
## EDA conclusion

The dataset contains short, informal Roman Urdu text and a noticeable class
imbalance. Phase 2 should use a stratified split, compare macro-F1 as well as
accuracy, and consider word and character n-grams because spellings vary.
"""
    ),
]


preprocessing_cells = [
    markdown(
        """
## Purpose

This notebook creates model-ready train and test CSV files without fitting a model.
It applies conservative text preprocessing, encodes the target, and uses a
reproducible stratified split.
"""
    ),
    common_setup,
    code(
        """
import re

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_PROCESSED / "roman_urdu_cleaned.csv")
print("Input shape:", df.shape)
df.head()
"""
    ),
    markdown(
        """
## 1. Prepare model text

We lowercase text, replace web links and usernames with reusable tokens, and make
spaces consistent. We do **not** remove punctuation, stem words, or delete common
words because those choices can remove sentiment information and are especially
risky for informal Roman Urdu.
"""
    ),
    code(
        """
def prepare_model_text(text):
    # Create a conservative model-ready version of one sentence.
    text = str(text).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " <url> ", text)
    text = re.sub(r"@\\w+", " <user> ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


df["model_text"] = df["sentence"].apply(prepare_model_text)
df[["sentence", "model_text"]].head(10)
"""
    ),
    markdown("## 2. Encode the target labels"),
    code(
        """
# The mapping is written explicitly so it will stay the same in Phase 2.
label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
df["label"] = df["sentiment"].map(label_mapping)

print("Label mapping:", label_mapping)
print("Missing encoded labels:", df["label"].isna().sum())
df[["sentiment", "label"]].drop_duplicates().sort_values("label")
"""
    ),
    markdown(
        """
## 3. Split into train and test sets

We use 80% for training and 20% for testing. `stratify` keeps approximately the
same sentiment percentages in both sets. `random_state=42` makes the split exactly
reproducible. The test set must remain untouched during Phase 2 model training.
"""
    ),
    code(
        """
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
"""
    ),
    code(
        """
def class_percentage(frame):
    return (frame["sentiment"].value_counts(normalize=True) * 100).round(2)


distribution_check = pd.DataFrame(
    {
        "full_data_%": class_percentage(df),
        "train_%": class_percentage(train_df),
        "test_%": class_percentage(test_df),
    }
)
distribution_check
"""
    ),
    markdown(
        """
## 4. Why numeric scaling is not applied

The main input is text, not a continuous measurement. StandardScaler or MinMaxScaler
cannot be applied directly to sentences. In Phase 2:

- TF-IDF will create and normalize sparse numeric features for classical models;
- an LSTM/transformer will tokenize text and learn embeddings;
- if hand-built numeric length features are added to a model, a scaler must be fit
  on the training set only and then applied to the test set.

Doing that later prevents data leakage and keeps this Phase 1 output reusable.
"""
    ),
    markdown("## 5. Save processed files and run leakage checks"),
    code(
        """
processed_path = DATA_PROCESSED / "roman_urdu_processed.csv"
train_path = DATA_PROCESSED / "train.csv"
test_path = DATA_PROCESSED / "test.csv"

df.to_csv(processed_path, index=False, encoding="utf-8")
train_df.to_csv(train_path, index=False, encoding="utf-8")
test_df.to_csv(test_path, index=False, encoding="utf-8")

train_texts = set(train_df["model_text"])
test_texts = set(test_df["model_text"])

print("Missing values in processed data:", int(df.isna().sum().sum()))
print("Duplicate model texts:", int(df["model_text"].duplicated().sum()))
print("Texts appearing in both train and test:", len(train_texts.intersection(test_texts)))
print("Saved:", processed_path)
print("Saved:", train_path)
print("Saved:", test_path)
"""
    ),
    markdown(
        """
## Preprocessing conclusion

The saved files contain readable sentiment names, fixed numeric labels, and a
conservative model-ready text column. The stratified split is reproducible and has
no repeated normalized text shared between train and test.
"""
    ),
]


save_notebook("01_data_cleaning.ipynb", "Phase 1 — Data Cleaning", cleaning_cells)
save_notebook("02_exploratory_data_analysis.ipynb", "Phase 1 — Exploratory Data Analysis", eda_cells)
save_notebook("03_preprocessing.ipynb", "Phase 1 — Preprocessing", preprocessing_cells)

print("Created notebooks in", NOTEBOOKS)

