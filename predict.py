"""
Check whether a URL looks like phishing, using the trained SVM.

Run:  py predict.py http://paypal-login.verify-account.xyz/signin.php
      py predict.py            (asks for URLs interactively)
"""

import sys

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, extract_features

model = joblib.load("svm_model.joblib")


def predict(url):
    X = pd.DataFrame([extract_features(url)], columns=FEATURE_COLUMNS)
    return "PHISHING" if model.predict(X)[0] == 1 else "LEGITIMATE"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for url in sys.argv[1:]:
            print(f"{predict(url):10}  {url}")
    else:
        while (url := input("Enter URL (blank to quit): ").strip()):
            print(" ->", predict(url))
