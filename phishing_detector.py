"""Train an email phishing detector from starter_phishing_dataset.csv."""
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "starter_phishing_dataset.csv"
MODEL_PATH = BASE_DIR / "phishing_detector.pkl"

SUSPICIOUS_WORDS = [
    "urgent", "verify", "password", "account", "suspended", "click",
    "bank", "login", "security", "confirm", "payment", "update",
    "winner", "free", "limited", "immediately", "expire", "claim",
    "credential", "invoice", "gift", "prize", "reset", "locked"
]


def preprocess_email(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s@]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_features(text):
    text = str(text)
    lower = text.lower()
    return [
        len(text),
        len(text.split()),
        text.count("!"),
        text.count("@"),
        text.count("."),
        text.count("/"),
        len(re.findall(r"https?://\S+|www\.\S+", text, re.I)),
        sum(char.isupper() for char in text),
        sum(lower.count(word) for word in SUSPICIOUS_WORDS),
    ]


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.name}. "
            "Put the CSV file next to phishing_detector.py."
        )

    data = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", low_memory=False)
    data.columns = [str(c).strip().lower() for c in data.columns]

    # This supplied dataset has a 'body' column and a numeric 'label' column.
    text_col = next((c for c in ["body", "text", "email", "message", "content"] if c in data.columns), None)
    label_col = next((c for c in ["label", "class", "target", "category"] if c in data.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not identify email and label columns. Found: {list(data.columns)}")

    data = data[[text_col, label_col]].dropna()
    data.columns = ["text", "label"]
    data["text"] = data["text"].astype(str)

    def convert_label(value):
        value = str(value).strip().lower()
        if value in {"0", "0.0", "legitimate", "ham", "safe", "benign", "normal"}:
            return 0
        if value in {"1", "1.0", "phishing", "spam", "scam", "fraud", "malicious"}:
            return 1
        return np.nan

    data["label"] = data["label"].apply(convert_label)
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both labels: 0 (legitimate) and 1 (phishing).")
    return data["text"], data["label"]


def main():
    print("\n=== AI PHISHING EMAIL DETECTOR: TRAINING ===")
    texts, labels = load_dataset()
    print(f"Loaded {len(texts)} emails. Label counts: {labels.value_counts().to_dict()}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_email,
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    x_train_text = vectorizer.fit_transform(x_train)
    x_test_text = vectorizer.transform(x_test)

    scaler = MaxAbsScaler()
    train_numeric = np.asarray([extract_features(t) for t in x_train], dtype=float)
    test_numeric = np.asarray([extract_features(t) for t in x_test], dtype=float)
    train_numeric = scaler.fit_transform(csr_matrix(train_numeric))
    test_numeric = scaler.transform(csr_matrix(test_numeric))

    x_train_all = hstack([x_train_text, train_numeric]).tocsr()
    x_test_all = hstack([x_test_text, test_numeric]).tocsr()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    rows = []
    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train_all, y_train)
        pred = model.predict(x_test_all)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
        })
        trained[name] = model

    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    results.to_csv(BASE_DIR / "model_comparison.csv", index=False)
    best_name = str(results.iloc[0]["Model"])
    best_model = trained[best_name]
    print("\nModel comparison:\n", results.to_string(index=False))
    print(f"\nSelected model: {best_name}")

    predictions = best_model.predict(x_test_all)
    print("\nClassification report:\n", classification_report(
        y_test, predictions, target_names=["Legitimate", "Phishing"], zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Phishing"]).plot(values_format="d")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # Save the model and the exact preprocessing objects used during training.
    joblib.dump({
        "model": best_model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "model_name": best_name,
        "feature_version": 1,
    }, MODEL_PATH)

    print("\nTraining finished.")
    print(f"Saved model: {MODEL_PATH}")
    print("Saved model_comparison.csv and confusion_matrix.png")
    print("Now run: python app.py")


if __name__ == "__main__":
    main()
