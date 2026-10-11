"""
Streamlit web app for phishing website detection.

Run:  streamlit run app.py      (run  py train.py  once first to create the model)
"""

import json

import joblib
import pandas as pd
import streamlit as st

from features import FEATURE_COLUMNS, extract_features

st.set_page_config(page_title="Phishing Website Detector", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("svm_model.joblib"), json.load(open("metrics.json"))


model, metrics = load_model()
rbf = metrics["results"]["SVM (RBF kernel)"]

with st.sidebar:
    st.header("📊 Dataset")
    st.metric("Unique URLs", f"{metrics['unique_rows']:,}")
    st.metric("Phishing", f"{metrics['phishing']:,}")
    st.metric("Legitimate", f"{metrics['legitimate']:,}")
    st.metric("URL features", metrics["n_features"])
    st.caption("Source: Vrbancic et al., Data in Brief (2020)")

st.title("🛡️ Phishing Website Detection")
st.caption("Support Vector Machine (RBF kernel) · URL-based features")

tab1, tab2 = st.tabs(["🔍 Check a URL", "📈 Model Performance"])

with tab1:
    st.subheader("Enter a website URL")
    url = st.text_input("URL", placeholder="e.g. http://secure-login.paypal.verify-account.xyz/signin.php")
    if st.button("Check", type="primary"):
        if not url.strip():
            st.warning("Please enter a URL first.")
        else:
            X = pd.DataFrame([extract_features(url)], columns=FEATURE_COLUMNS)
            pred = model.predict(X)[0]
            score = model.decision_function(X)[0]
            if pred == 1:
                st.error(f"🚨 **PHISHING** — SVM score {score:+.2f} (positive = phishing side)")
            else:
                st.success(f"✅ **LEGITIMATE** — SVM score {score:+.2f} (negative = legitimate side)")
            with st.expander("Show extracted URL features"):
                feats = X.T.rename(columns={0: "value"})
                st.dataframe(feats[feats["value"] != -1], height=300)

    st.divider()
    st.caption("Quick examples to try:")
    col1, col2 = st.columns(2)
    col1.code("http://paypal-login.verify-account.xyz/signin/update.php?user=a@b.com")
    col2.code("https://en.wikipedia.org/wiki/Support_vector_machine")

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{rbf['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{rbf['precision'] * 100:.2f}%")
    c3.metric("ROC AUC", f"{rbf['roc_auc']:.3f}")

    st.subheader("RBF vs Linear SVM")
    table = pd.DataFrame(metrics["results"]).drop("confusion_matrix").astype(float).round(4)
    st.dataframe(table)

    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png", width=380)
    st.subheader("ROC Curve")
    st.image("roc_curve.png", width=420)
