# ModelBuilding.py — 3-page Streamlit app (Welcome / Dashboard / Predictor)
# - English-only normalization
# - Trains many linear baselines (TF-IDF word+char)
# - Caches artifacts to ./model_cache so you don't retrain every edit
# - Dashboard: leaderboard, confusion matrix, per-class F1, learning curve (Train vs Test)
# - Predictor: probability bar chart (when model exposes probabilities)
# streamlit run ModelBuilding.py

import warnings
warnings.filterwarnings("ignore")

import os, re, sys, hashlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn

from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score
)

import streamlit as st

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

# ---------------- CONFIG ----------------
CACHE_VERSION = 11                     # bump to refresh Streamlit in-memory cache
DATA_PATH = r"C:\Users\Kai\Downloads\FYP Coding\CleanCombineData_featured.csv"
PAGES = ["Welcome", "Dashboard", "Predictor"]

# Disk cache (survives code changes)
MODEL_CACHE_DIR = Path("./model_cache")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Bump when you change training logic, features, or hyperparams
MODEL_CODE_TAG = "v5-eng-only-tfidf(word+char)-oversample-top3-learningcurve"


# -------- Column detection (skip obvious metadata) --------
LIKELY_META = {"id","ids","index","idx","user","username","name","handle","screen_name",
               "source","orig_source","platform","lang","language","country",
               "date","time","timestamp","created_at","url","link"}

def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    obj_cols = [c for c in cols if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError("No object/string columns found. Ensure text/label are string columns.")

    # label candidates: 2..20 uniq values, avoid metadata
    cand = [(c, df[c].nunique()) for c in obj_cols
            if 2 <= df[c].nunique() <= 20 and c.lower() not in LIKELY_META]
    cand.sort(key=lambda x: x[1])
    hints = ["label","sentiment","status","mental","category","tag","risk","class"]
    hinted = [c for c in cand if any(k in c[0].lower() for k in hints)]
    label_col = hinted[0][0] if hinted else (cand[0][0] if cand else None)

    # text: longest avg length, prefer common names
    def avglen(series): return series.dropna().astype(str).str.len().mean()
    text_cands = [c for c in obj_cols if c.lower() not in LIKELY_META]
    text_col = max(text_cands, key=lambda c: avglen(df[c]))
    for c in text_cands:
        if any(k in c.lower() for k in ["text","clean","content","message","post","tweet","comment"]) and avglen(df[c]) >= 10:
            text_col = c; break

    if not text_col or not label_col:
        raise ValueError(f"Auto-detect failed. Objects: {obj_cols}. Provide explicit names.")
    return text_col, label_col

# ---------------- English-only normalizer ----------------
REPEAT_CHARS = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = REPEAT_CHARS.sub(r"\1\1", s)                              # soooo → soo
    s = re.sub(r"[^\w#@_!?'\s\-]", " ", s, flags=re.UNICODE)      # keep word chars and a few symbols
    toks = [t.lower() for t in s.split()]
    return " ".join(toks)

class TextNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return [normalize_text(x) for x in X]

# ---------------- Oversample for macro-F1 ----------------
def oversample_train(X, y, seed=42):
    rng = np.random.default_rng(seed)
    counts = Counter(y); maxc = max(counts.values())
    idxs = np.arange(len(y)); keep = []
    for cls, cnt in counts.items():
        cls_idx = idxs[y == cls]
        reps = maxc // cnt
        rem  = maxc - reps*cnt
        up = np.concatenate([cls_idx]*reps + ([rng.choice(cls_idx, size=rem, replace=True)] if rem>0 else []))
        keep.append(up)
    keep = np.concatenate(keep); rng.shuffle(keep)
    return X[keep], y[keep]

# ---------------- Features: TF-IDF (word + char) ----------------
def build_features(char_weight=1.6):
    word = ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,3),
                                    min_df=1, max_df=0.95, sublinear_tf=True))
    char = ("char", TfidfVectorizer(analyzer="char", ngram_range=(3,6),
                                    min_df=1, max_df=0.95, sublinear_tf=True))
    return FeatureUnion([word, char], transformer_weights={"word": 1.0, "char": char_weight})

# ---------------- Train all models ----------------
def compute_all_models(csv_path: str, seed: int = 42):
    df = pd.read_csv(csv_path)
    text_col, label_col = detect_columns(df)

    data = df[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str).str.strip()
    data = data[data[text_col].str.len() > 0]

    le = LabelEncoder()
    y_full = le.fit_transform(data[label_col].values)
    X_full = data[text_col].values
    classes = list(le.classes_)
    print(f"[INFO] Using text='{text_col}' | label='{label_col}'")
    print(f"[INFO] Rows={len(data)} | Classes={classes}")

    Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full)
    Xtr_bal, ytr_bal = oversample_train(np.array(Xtr), np.array(ytr))

    feats = build_features()
    models = {
        "LinearSVM_C0.5": LinearSVC(class_weight="balanced", C=0.5),
        "LinearSVM_C1.0": LinearSVC(class_weight="balanced", C=1.0),
        "LinearSVM_C2.0": LinearSVC(class_weight="balanced", C=2.0),
        "LogReg_C1.0": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", C=1.0),
        "LogReg_C2.0": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", C=2.0),
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB(),
        "SGD_Hinge": SGDClassifier(loss="hinge", class_weight="balanced", max_iter=2000),
        "SGD_Log":   SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000),
        "PassiveAggressive": PassiveAggressiveClassifier(class_weight="balanced", max_iter=2000),
        "RidgeClassifier":   RidgeClassifier(class_weight="balanced")
    }

    results = []
    for name, clf in models.items():
        pipe = Pipeline([("norm", TextNormalizer()), ("feats", feats), ("clf", clf)])
        print(f"\n[TRAIN] {name} ...")
        pipe.fit(Xtr_bal, ytr_bal)
        yhat = pipe.predict(Xte)

        acc   = accuracy_score(yte, yhat)
        f1m   = f1_score(yte, yhat, average="macro")
        precm = precision_score(yte, yhat, average="macro", zero_division=0)
        recm  = recall_score(yte, yhat, average="macro", zero_division=0)

        repd = classification_report(yte, yhat, target_names=classes, zero_division=0, output_dict=True)
        cm   = confusion_matrix(yte, yhat)
        results.append({
            "name": name,
            "f1_macro": f1m, "accuracy": acc,
            "precision_macro": precm, "recall_macro": recm,
            "report": repd, "cm": cm, "pipe": pipe, "yhat": yhat
        })

    results.sort(key=lambda d: d["f1_macro"], reverse=True)
    return results, classes, np.array(Xtr_bal), np.array(ytr_bal), np.array(Xte), np.array(yte), X_full, y_full

# ---------------- Learning curve (Train vs Test) ----------------
def precompute_learning_curves(top_results: list, X: np.ndarray, y: np.ndarray, seed: int = 42):
    curves = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    train_sizes = np.linspace(0.1, 1.0, 5)

    for r in top_results:
        pipe = r["pipe"]
        sizes, train_scores, test_scores = learning_curve(
            pipe, X, y, cv=cv, scoring="f1_macro",
            train_sizes=train_sizes, n_jobs=-1, shuffle=True, random_state=seed
        )
        curves[r["name"]] = {
            "sizes": sizes.tolist(),
            "train": train_scores.mean(axis=1).tolist(),
            "test":  test_scores.mean(axis=1).tolist()
        }
    return curves

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

# ---------------- Disk cache helpers ----------------
def _dataset_fingerprint(csv_path: str, sample_rows: int = 5000) -> str:
    p = Path(csv_path)
    if not p.exists():
        return "nofile"
    stat = p.stat()
    meta = f"{p.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    try:
        df_sample = pd.read_csv(csv_path, nrows=sample_rows, dtype=str)
        payload = meta + "|" + str(df_sample.head(50).to_dict())
    except Exception:
        payload = meta
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _model_cache_key(csv_path: str) -> str:
    fp = _dataset_fingerprint(csv_path)
    py  = f"py{sys.version_info.major}.{sys.version_info.minor}"
    skl = f"skl{sklearn.__version__}"
    return f"{fp}-{MODEL_CODE_TAG}-{py}-{skl}"

def _cache_path_for(key: str) -> Path:
    return MODEL_CACHE_DIR / f"{key}.joblib"

def load_from_disk_cache(csv_path: str):
    key  = _model_cache_key(csv_path)
    path = _cache_path_for(key)
    if path.exists():
        try:
            payload = joblib.load(path)
            print(f"[CACHE] Loaded artifacts from {path.name}")
            return payload
        except Exception as e:
            print(f"[CACHE] Failed to load {path.name}: {e}")
    return None

def save_to_disk_cache(csv_path: str, payload: dict):
    key  = _model_cache_key(csv_path)
    path = _cache_path_for(key)
    try:
        joblib.dump(payload, path, compress=("xz", 3))
        print(f"[CACHE] Saved model artifacts → {path}")
    except Exception as e:
        print(f"[CACHE] Save failed: {e}")

# ---------------- One-shot cached build (train + precompute Top-3 curves) ----------------
@st.cache_resource(show_spinner=True)
def build_app_data(csv_path: str, _ui_version: int):
    cached = load_from_disk_cache(csv_path)
    if cached is not None:
        return cached

    results, classes, Xtr_bal, ytr_bal, Xte, yte, X_full, y_full = compute_all_models(csv_path)
    top3 = results[:min(3, len(results))]
    curves_top3 = precompute_learning_curves(top3, X_full, y_full, seed=42)

    leaderboard = pd.DataFrame(
        [{
            "Rank": i+1,
            "Model": r["name"],
            "Macro-F1": r["f1_macro"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision_macro"],
            "Recall": r["recall_macro"],
        } for i, r in enumerate(results)]
    )

    payload = {
        "results": results,
        "classes": classes,
        "Xtr_bal": Xtr_bal, "ytr_bal": ytr_bal,
        "Xte": Xte, "yte": yte,
        "leaderboard": leaderboard,
        "curves_top3": curves_top3,
        "best_pipe": results[0]["pipe"],
        "meta": {
            "dataset_path": str(csv_path),
            "dataset_fp": _dataset_fingerprint(csv_path),
            "sklearn": sklearn.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "code_tag": MODEL_CODE_TAG,
        }
    }
    save_to_disk_cache(csv_path, payload)
    return payload

# ---------------- Plot helpers (beautified) ----------------
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

# ---------- Probability utilities ----------
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
    dfp = pd.DataFrame({"Class": classes, "Probability": proba})
    dfp = dfp.sort_values("Probability", ascending=True)  # for horizontal plot

    base = np.array([0.45, 0.67, 0.89, 1.0])  # soft blue
    muted = np.array([0.80, 0.86, 0.93, 1.0]) # light bar
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

# ---------------- UI: CSS & Nav ----------------
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
    """Top pill button bar to switch pages and keep URL in sync; defaults to Welcome."""
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
        st.button("🏠  Welcome", use_container_width=True,
                  on_click=lambda: go("Welcome"), key="btn_welcome")
    with c2:
        st.button("📊  Dashboard", use_container_width=True,
                  on_click=lambda: go("Dashboard"), key="btn_dashboard")
    with c3:
        st.button("🔎  Predictor", use_container_width=True,
                  on_click=lambda: go("Predictor"), key="btn_predictor")
    st.markdown("---")

# ---------------- UI Pages ----------------
def page_welcome():
    st.title("Mental Health Sentiment — Research Prototype")

    st.markdown(
        """
        ### What this app does
        - Trains a suite of classic linear text classifiers (**LogReg**, **LinearSVM**, **SGD**, **Passive-Aggressive**, **Ridge**, **Naive Bayes**) on *your* dataset.  
        - Uses a lightweight **English-only** preprocessor and **TF-IDF** features (**word 1–3-grams** + **char 3–6-grams**).  
        - Handles class imbalance via **simple oversampling** on the training split.  
        - Compares models with **Accuracy**, **Macro-Precision**, **Macro-Recall**, and **Macro-F1** (*primary*).  
        - Precomputes **learning curves** (Train vs Test Macro-F1) for the **Top-3** models.
        """
    )

    st.markdown(
        """
        ### How to use
        1. Open **Dashboard** → review the leaderboard, confusion matrix, per-class F1, and learning curve.  
        2. Use the selector to switch among the **Top-3** models.  
        3. Go to **Predictor** → type English text and view the predicted label with a probability bar chart.
        """
    )

    st.markdown(
        f"""
        ### Under the hood
        - **Vectorizer:** TF-IDF (word + char), sublinear TF; simple normalizer.  
        - **Split:** 80/20 stratified train/test; oversampling only on train.  
        - **Caching:** Model artifacts saved to `./model_cache/` and reused across runs.  
        - **Dataset path:** `{DATA_PATH}`
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

    with st.expander("Limitations & scope"):
        st.markdown(
            """
            - **English-only** prototype; not a clinical tool.  
            - Small or imbalanced datasets can make metrics unstable—use the learning curve to judge bias/variance.  
            """
        )

    with st.expander("Reproducibility & re-training"):
        st.markdown(
            """
            - To **force retrain**, delete files in `model_cache/` *or* bump `MODEL_CODE_TAG` in this script.  
            - Prebuild without UI:
              ```bash
              python app_sentiment.py --precompute
              ```
            - Record for your FYP: Python & scikit-learn versions, TF-IDF settings, class rebalance method,
              random seeds, and fixed train/test indices.  
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
    top3 = data["results"][:min(3, len(data["results"]))]
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

    # Misclassified examples
    st.subheader("Sample Misclassifications")
    ytrue = pd.Series([data["classes"][i] for i in data["yte"]], name="true")
    ypred = pd.Series([data["classes"][i] for i in r["yhat"]], name="pred")
    mis = (ytrue.values != ypred.values)
    st.caption(f"Showing up to 100 misclassified examples (of {mis.sum()} total).")
    st.dataframe(pd.DataFrame({"text": data["Xte"][mis], "true": ytrue[mis].values, "pred": ypred[mis].values}).head(100),
                 use_container_width=True)

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

# ---------------- Main App ----------------
def main():
    st.set_page_config(page_title="Sentiment App", page_icon="📊", layout="wide")
    inject_css()

    with st.spinner("Preparing models and dashboard (first run only)…"):
        data = build_app_data(DATA_PATH, CACHE_VERSION)

    # Top navigation
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Welcome"
    render_top_nav()

    page = st.session_state.get("nav", "Welcome")
    if page == "Welcome":
        page_welcome()
    elif page == "Dashboard":
        page_dashboard(data)
    else:
        page_predictor(data)

if __name__ == "__main__":
    if "--precompute" in sys.argv:
        _ = build_app_data(DATA_PATH, CACHE_VERSION)
        print("[DONE] Precomputed and cached models.")
    else:
        main()
