import argparse
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
import joblib
from bixie.models.model_inference import SAFEEmbedder, CLASSIFIER_PATH

def load_labeled_samples(data_dir: Path) -> List[Dict]:
    """
    Traverse the data directory and collect labeled samples.
    Labels are inferred from filenames: 'good' = safe, 'bad' = vulnerable.
    """
    samples = []
    for file in data_dir.rglob("*.*"):
        if not file.is_file():
            continue
        fname = file.name.lower()
        if "bad" in fname:
            label = "vulnerable"
        elif "good" in fname:
            label = "good"
        else:
            continue
        samples.append({"path": file, "label": label})
    return samples

def extract_embeddings(samples: List[Dict], embedder: SAFEEmbedder):
    X, y, failed = [], [], 0
    for sample in samples:
        emb = embedder.embed_file(sample["path"])
        if emb is not None:
            X.append(emb)
            y.append(1 if sample["label"] == "vulnerable" else 0)
        else:
            failed += 1
    return np.array(X), np.array(y), failed

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Vulnerable"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train and benchmark Bixie model on labeled samples.")
    parser.add_argument("--data", type=str, required=True, help="Path to data folder (Juliet or similar)")
    parser.add_argument("--output", type=str, default="benchmark_report.txt", help="Output report file")
    parser.add_argument("--confusion", type=str, default="confusion_matrix.png", help="Confusion matrix image file")
    parser.add_argument("--save-model", action="store_true", help="Save trained classifier to disk")
    args = parser.parse_args()

    data_dir = Path(args.data)
    samples = load_labeled_samples(data_dir)
    if not samples:
        print("No labeled samples found.")
        return

    print(f"Loaded {len(samples)} samples from {data_dir}")

    embedder = SAFEEmbedder()
    X, y, failed = extract_embeddings(samples, embedder)
    print(f"Extracted {len(X)} embeddings ({failed} failed)")

    if len(X) < 2:
        print("Not enough samples for training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Good", "Vulnerable"])

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    with open(args.output, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(report)

    plot_confusion(y_test, y_pred, args.confusion)
    print(f"Confusion matrix saved to {args.confusion}")

    if args.save_model:
        joblib.dump(clf, CLASSIFIER_PATH)
        print(f"Trained classifier saved to {CLASSIFIER_PATH}")
