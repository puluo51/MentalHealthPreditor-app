# dashboard.py
# ------------
# 3-page Streamlit app using artifacts from modelbuilding.py
# Run: streamlit run dashboard.py

import warnings
warnings.filterwarnings("ignore")

import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline  # for type hints only

from modelbuilding import build_app_data  # our training + cache module

# ---------- GLOBAL MATPLOTLIB STYLE ----------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.autolayout": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelweight": "semibold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

# ---------- APP CONFIG ----------
CACHE_VERSION = 11                      # UI cache/version bump
PAGES = ["Welcome", "Dashboard", "Predictor"]

# Set your dataset path here
DATA_PATH = r"C:\Users\Kai\Downloads\FYP Coding\CleanCombineData_featured.csv"

# ---------- Plot helpers ----------
def per_class_f1(report_dict, classes):
    return np.array([report_dict.get(c, {}).get("f1-score", np.nan) for c in classes])

def plot_confusion(cm, classes, title):
    fig = plt.figure(figsize=(6.6, 5.4))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > 0.6 * vmax else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=10)
    return fig

def plot_f1_bars(report_dict, classes, title):
    vals = per_class_f1(report_dict, classes)
    order = np.argsort(vals)
    fig = plt.figure(figsize=(6.6, 5.0))
    y = np.arange(len(classes))
    plt.barh(y, vals[order], height=0.6)
    plt.yticks(y, [classes[i] for i in order])
    plt.xlim(0, 1.0)
    plt.xlabel("F1-score"); plt.title(title)
    for i, v in enumerate(vals[order]):
        plt.text(v + 0.01 if v < 0.95 else 0.95, i, f"{v:.2f}",
                 va="center", ha="left" if v < 0.95 else "right")
    return fig

def plot_learning_curve(curve: dict, title="Learning Curve (Macro-F1)"):
    sizes = curve["sizes"]
    fig = plt.figure(figsize=(7, 4.4))
    plt.plot(sizes, curve["train"], marker="o", linewidth=2.2, label="Train")
    plt.plot(sizes, curve["test"],  marker="o", linewidth=2.2, label="Test (CV)")
    plt.xlabel("Training examples")
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True)
    return fig

def _class_probabilities(pipe: Pipeline, text: str):
    """Return per-class probabilities; fallback to softmax(decision_function)."""
    try:
        return pipe.predict_proba([text])[0]
    except Exception:
        try:
            scores = pipe.decision_function([text])
            s = np.asarray(scores)[0]
            if np.ndim(s) == 0:  # binary
                s = np.array([-s, s])
            e = np.exp(s - np.max(s))
            return e / e.sum()
        except Exception:
            return None

def plot_probability_bars(classes, proba, predicted_index):
    dfp = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values("Probability")
    base = np.array([0.45, 0.67, 0.89, 1.0])   # soft blue
    muted = np.array([0.80, 0.86, 0.93, 1.0])  # light
    colors = [tuple(base) if classes[predicted_index]==cls else tuple(muted) for cls in dfp["Class"]]
    fig = plt.figure(figsize=(7.6, 5))
    y = np.arange(len(dfp))
    plt.barh(y, dfp["Probability"], color=colors, edgecolor="#333333")
    plt.yticks(y, dfp["Class"])
    plt.xlim(0, 1.0)
    plt.xlabel("Probability")
    plt.title("Class probabilities")
    for i, v in enumerate(dfp["Probability"]):
        plt.text(v + 0.01 if v < 0.95 else 0.95, i, f"{v:.3f}",
                 va="center", ha="left" if v < 0.95 else "right", fontsize=10)
    return fig

# ---------- UI Chrome ----------
def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }
        h1, h2 { letter-spacing: .3px; }
        h1 { font-weight: 800; }  h2 { font-weight: 700; }
        .pill-row { margin: 8px 0 12px 0; display: flex; gap: 8px; }
        .pill-row button[kind="primary"] {
            border-radius: 999px !important; padding: .55rem 1.1rem !important; font-weight: 700 !important;
        }
        .pill-row button { border-radius: 999px !important; padding: .55rem 1.1rem !important; }
        .stAlert > div { border-radius: 12px; }
        .stDataFrame table thead th { font-weight: 700; }
        </style>
        """, unsafe_allow_html=True
    )

def render_top_nav():
    """Pill buttons + query param sync; default to Welcome."""
    if "nav" not in st.session_state or st.session_state.get("_nav_ver") != CACHE_VERSION:
        st.session_state["nav"] = "Welcome"
        st.session_state["_nav_ver"] = CACHE_VERSION
        st.query_params["page"] = "Welcome"

    qs_page = st.query_params.get("page")
    if isinstance(qs_page, list):
        qs_page = qs_page[0] if qs_page else None
    if isinstance(qs_page, str) and qs_page in PAGES:
        st.session_state["nav"] = qs_page

    def go(page_name: str):
        st.session_state["nav"] = page_name
        st.query_params["page"] = page_name

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.button("🏠  Welcome", use_container_width=True, on_click=lambda: go("Welcome"), key="btn_welcome")
    with c2:
        st.button("📊  Dashboard", use_container_width=True, on_click=lambda: go("Dashboard"), key="btn_dashboard")
    with c3:
        st.button("🔎  Predictor", use_container_width=True, on_click=lambda: go("Predictor"), key="btn_predictor")
    st.markdown("---")

# ---------- Pages ----------
def page_welcome(data: Dict[str, Any]):
    st.title("Mental Health Sentiment — Research Prototype")

    st.markdown(
        """
        ### What this app does
        - Trains classic linear text classifiers (**LogReg**, **LinearSVM**, **SGD**, **Passive-Aggressive**, **Ridge**, **Naive Bayes**) on *your* dataset.  
        - Uses a lightweight **English-only** preprocessor and **TF-IDF** features (**word 1–3-grams** + **char 3–6-grams**).  
        - Handles class imbalance via **simple oversampling** on the training split.  
        - Compares models with **Accuracy**, **Macro-Precision**, **Macro-Recall**, and **Macro-F1** (*primary*).  
        - Precomputes **learning curves** (Train vs Test Macro-F1) for the **Top-3** models.
        """
    )

    st.markdown(
        """
        ### How to use
        1. Open **Dashboard** → review leaderboard, confusion matrix, per-class F1, and learning curve.  
        2. Use the selector to switch among the **Top-3** models.  
        3. Open **Predictor** → type English text and view predicted label + probability bar chart.
        """
    )

    st.markdown(
        f"""
        ### Under the hood
        - **Vectorizer:** TF-IDF (word + char), sublinear TF; simple normalizer.  
        - **Split:** 80/20 stratified train/test; oversampling only on train.  
        - **Caching:** Artifacts saved to `./model_cache/` and reused.  
        - **Dataset path:** `{data['meta']['dataset_path']}`
        """
    )

    with st.expander("Metric cheat-sheet"):
        st.markdown(
            """
            - **Accuracy** — overall fraction of correct predictions.  
            - **Macro-Precision** — average precision across classes.  
            - **Macro-Recall** — average recall across classes.  
            - **Macro-F1** — average F1 across classes; our main ranking metric.  
            """
        )

def page_dashboard(data: Dict[str, Any]):
    st.title("Sentiment Model Dashboard (Top-3)")
    st.subheader("Leaderboard")
    styled = data["leaderboard"].style.format(
        {"Macro-F1": "{:.4f}", "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}
    ).background_gradient(axis=0, subset=["Macro-F1"], cmap="Greens")
    st.dataframe(styled, use_container_width=True)

    # Tabs for visuals
    top3 = data["results"][:min(3, len(data["results"]))]  # keep in case you add more models later
    names = [r["name"] for r in top3]
    pick = st.selectbox("Select a top model to inspect", options=names, index=0)
    r = next(x for x in top3 if x["name"] == pick)

    t1, t2, t3 = st.tabs(["Confusion Matrix", "Per-class F1", "Learning Curve"])
    with t1:
        st.pyplot(plot_confusion(r["cm"], data["classes"], f"Confusion | {r['name']}"), clear_figure=True)
    with t2:
        st.pyplot(plot_f1_bars(r["report"], data["classes"], f"Per-class F1 | {r['name']}"), clear_figure=True)
    with t3:
        curve = data["curves_top3"].get(r["name"])
        if curve is not None:
            st.pyplot(plot_learning_curve(curve, title=f"Learning Curve | {r['name']}"), clear_figure=True)
        else:
            st.info("Learning curve not available for this model.")

    # Misclassified examples (up to 100)
    st.subheader("Sample Misclassifications")
    ytrue = pd.Series([data["classes"][i] for i in data["yte"]], name="true")
    ypred = pd.Series([data["classes"][i] for i in r["yhat"]], name="pred")
    mis = (ytrue.values != ypred.values)
    st.caption(f"Showing up to 100 misclassified examples (of {mis.sum()} total).")
    st.dataframe(
        pd.DataFrame({"text": data["Xte"][mis], "true": ytrue[mis].values, "pred": ypred[mis].values}).head(100),
        use_container_width=True
    )

def page_predictor(data: Dict[str, Any]):
    st.title("User Input Predictor (English Only)")
    st.caption("Uses the **best model** from the dashboard (same preprocessing & features).")

    txt = st.text_area(
        "Enter your text",
        height=180,
        placeholder="e.g., I feel extremely hopeless today and can't sleep at all..."
    )
    if st.button("Predict", use_container_width=True) and txt.strip():
        pipe    = data["best_pipe"]
        classes = data["classes"]

        pred = pipe.predict([txt])[0]
        label = classes[pred] if isinstance(pred, (int, np.integer)) and 0 <= pred < len(classes) else str(pred)
        st.success(f"Prediction: **{label}**")

        proba = _class_probabilities(pipe, txt)
        if proba is not None and len(proba) == len(classes):
            fig = plot_probability_bars(classes, proba, predicted_index=int(pred))
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("This model doesn’t expose probabilities; showing only the predicted label.")

# ---------- Main ----------
def main():
    st.set_page_config(page_title="Sentiment App", page_icon="📊", layout="wide")
    inject_css()

    with st.spinner("Preparing models and dashboard (first run only)…"):
        data = build_app_data(DATA_PATH)

    render_top_nav()
    page = st.session_state.get("nav", "Welcome")
    if page == "Welcome":
        page_welcome(data)
    elif page == "Dashboard":
        page_dashboard(data)
    else:
        page_predictor(data)

if __name__ == "__main__":
    main()
