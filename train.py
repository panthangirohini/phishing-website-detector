"""
Train an SVM to detect phishing websites from URL features.

Run:  py train.py
Creates: svm_model.joblib, metrics.json, confusion_matrix.png, roc_curve.png
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from features import FEATURE_COLUMNS

# 1. Load data, keep only the URL features, drop duplicate rows
#    (over half the rows repeat, which would leak test rows into training)
df = pd.read_csv("dataset_small.csv")
raw_rows = len(df)
df = df[FEATURE_COLUMNS + ["phishing"]].drop_duplicates()
print(f"Rows: {raw_rows} -> {len(df)} after removing duplicates")

X, y = df[FEATURE_COLUMNS], df["phishing"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Train two SVMs: RBF kernel (main model) and linear (baseline)
models = {
    "SVM (RBF kernel)": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
    "SVM (linear)": make_pipeline(
        StandardScaler(), LinearSVC(C=1, max_iter=20000, class_weight="balanced")),
}

results = {}
plt.figure(1, figsize=(5.5, 4.2))
for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    fpr, tpr, _ = roc_curve(y_test, scores)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
    print(f"  accuracy = {results[name]['accuracy']:.4f}")

# 3. Plots
plt.figure(1)
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve (test set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)

best = models["SVM (RBF kernel)"]
fig, ax = plt.subplots(figsize=(5, 4.2))
ConfusionMatrixDisplay.from_estimator(
    best, X_test, y_test, display_labels=["Legitimate", "Phishing"],
    cmap="Blues", values_format="d", colorbar=False, ax=ax)
ax.set_title("Confusion matrix - SVM (RBF)")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=200)

# 4. Save model and metrics
joblib.dump(best, "svm_model.joblib")
json.dump({
    "raw_rows": raw_rows, "unique_rows": len(df), "n_features": len(FEATURE_COLUMNS),
    "train_rows": len(X_train), "test_rows": len(X_test),
    "phishing": int(y.sum()), "legitimate": int((y == 0).sum()),
    "results": results,
}, open("metrics.json", "w"), indent=2)
print("Saved svm_model.joblib, metrics.json, confusion_matrix.png, roc_curve.png")
