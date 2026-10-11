# Phishing Website Detection Using Support Vector Machine

A machine learning mini-project that classifies a website URL as **Phishing** or **Legitimate** using a Support Vector Machine. It has a console version and a web GUI.

## Overview

| | |
|---|---|
| **Algorithm** | Support Vector Machine (RBF kernel), compared with a Linear SVM |
| **Dataset** | Vrbančič et al., *Datasets for phishing websites detection* (2020), `dataset_small.csv` (58,645 rows) |
| **Features** | 98 lexical URL features (character counts and lengths of the URL, domain, directory, file and query) |
| **Interfaces** | Console scripts **and** a Streamlit web app |
| **Language / Libraries** | Python, pandas, scikit-learn, streamlit, matplotlib |

## Project Structure

```
phishing-website-detector/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── dataset_small.csv          # Dataset
├── features.py                # Turns a URL into the 98 features
├── train.py                   # Trains and evaluates the SVMs, saves the model and plots
├── predict.py                 # Console: classify URLs typed by the user
├── app.py                     # Web GUI (Streamlit)
├── svm_model.joblib           # Trained model (created by train.py)
├── metrics.json               # Test scores (created by train.py)
├── confusion_matrix.png       # Plot (created by train.py)
└── roc_curve.png              # Plot (created by train.py)
```

The trained model is already included, so the web app and `predict.py` work straight away. Run `train.py` only to retrain.

## Setup

1. Install Python 3.8 or newer.
2. Open this folder in VS Code (or any terminal).
3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

## Run

**1. Train the model** (optional, takes about 1–2 minutes):

```
python train.py
```

**2. Web app** (recommended for presentation demos):

```
streamlit run app.py
```

This opens a browser tab with:
- **Check a URL** tab: type any URL and get an instant Phishing or Legitimate result with the SVM score. You can also expand the extracted features.
- **Model Performance** tab: accuracy, precision, ROC AUC, the RBF vs Linear comparison table, the confusion matrix and the ROC curve.
- **Sidebar**: dataset statistics.

**3. Console version:**

```
python predict.py http://paypal-login.verify-account.xyz/signin.php
python predict.py          # asks for URLs one by one
```

> On Windows, use `py` instead of `python` if `python` is not recognised.

## How It Works

1. **Dataset loading**: reads the CSV and keeps the 98 columns that can be computed from the URL text alone.
2. **Duplicate removal**: 54% of rows are exact copies once only URL features are kept. They are removed (58,645 → 26,902 rows) so the test set contains no rows seen in training.
3. **Train-test split**: 80% for training and 20% for testing, stratified by class.
4. **Feature scaling**: `StandardScaler` gives every feature zero mean and unit variance, because SVMs depend on distances.
5. **Model training**: an RBF-kernel `SVC` (C = 10, gamma = "scale") with `class_weight="balanced"`, since phishing rows outnumber legitimate ones about 4 to 1. A `LinearSVC` is trained as a baseline.
6. **Evaluation**: accuracy, precision, recall, F1, ROC AUC and the confusion matrix on the test set.
7. **Prediction**: `features.py` converts any new URL into the same 98 features, so the model can classify it instantly without visiting the website.

## Results (test set: 5,381 URLs)

| Metric | SVM (RBF) | SVM (Linear) |
|---|---|---|
| Accuracy | 80.23% | 79.63% |
| Precision | 95.68% | 94.67% |
| Recall | 79.18% | 79.34% |
| F1-score | 86.65% | 86.33% |
| ROC AUC | 0.894 | 0.883 |

## Why an SVM works well here

An SVM finds the boundary that separates the two classes with the widest possible margin, which makes it generalise well. With the RBF kernel it can draw curved boundaries, so it can capture combinations of features, such as a long URL with many hyphens and an `@` sign, that a straight-line model misses.
