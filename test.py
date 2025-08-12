# dashboard.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# -------------------------
# Project paths (your layout)
# -------------------------
ROOT = Path(".")
ML_PATH = ROOT / "results" / "ml_models"
DL_PATH = ROOT / "results" / "dl_models"
ENSEMBLE_PATH = ROOT / "results" / "ensemble_models"
RULE_PATH = ROOT / "results" / "rule_based_models"

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="HMPV Dashboard", layout="wide", page_icon="📊")
st.title("HMPV — Sentiment & Emotion Dashboard")
st.markdown("Minimal presentation view. Use the **sidebar** for real-time predictions (per-model + ensemble).")

# -------------------------
# Utilities
# -------------------------
def safe_joblib_load(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

def safe_h5_load(path: Path):
    if not path.exists():
        return None
    try:
        return load_model(str(path))
    except Exception:
        return None

def ensure_text_col(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns and "comment" in df.columns:
        return df.rename(columns={"comment": "text"})
    return df

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    import nltk
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

def preprocess_ml(text: str):
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", "", t)
    tokens = [lemmatizer.lemmatize(w) for w in t.split() if w not in stop_words]
    return " ".join(tokens)

# rule-based helpers
try:
    vader = SentimentIntensityAnalyzer()
except Exception:
    import nltk
    nltk.download("vader_lexicon")
    vader = SentimentIntensityAnalyzer()

def vader_label(text: str, pos_thr=0.05, neg_thr=-0.05):
    s = vader.polarity_scores(str(text))
    c = s["compound"]
    if c >= pos_thr:
        return "1"
    if c <= neg_thr:
        return "-1"
    return "0"

def textblob_label(text: str, pos_thr=0.05, neg_thr=-0.05):
    p = TextBlob(str(text)).sentiment.polarity
    if p >= pos_thr:
        return "1"
    if p <= neg_thr:
        return "-1"
    return "0"

# decode helper: try to inverse_transform if encoder exists, otherwise return string
def decode_prediction(raw_pred, label_encoder):
    if label_encoder is None:
        return str(raw_pred)
    try:
        # If model returns integer/np.int64
        return label_encoder.inverse_transform([int(raw_pred)])[0]
    except Exception:
        try:
            # If model returns string-coded value already (like '-1','0','1')
            return label_encoder.inverse_transform([str(raw_pred)])[0]
        except Exception:
            # fallback
            return str(raw_pred)

# -------------------------
# Cached loaders
# -------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    # ML models + vectorizer + encoders
    artifacts["rf_sent"] = safe_joblib_load(ML_PATH / "randomforest_sentiment.joblib")
    artifacts["svm_sent"] = safe_joblib_load(ML_PATH / "svm_sentiment.joblib")
    artifacts["nb_sent"] = safe_joblib_load(ML_PATH / "naivebayes_sentiment.joblib")
    artifacts["rf_em"] = safe_joblib_load(ML_PATH / "randomforest_emotion.joblib")
    artifacts["svm_em"] = safe_joblib_load(ML_PATH / "svm_emotion.joblib")
    artifacts["nb_em"] = safe_joblib_load(ML_PATH / "naivebayes_emotion.joblib")
    artifacts["tfidf"] = safe_joblib_load(ML_PATH / "tfidf_vectorizer.joblib")
    artifacts["le_sent_ml"] = safe_joblib_load(ML_PATH / "label_encoder_sentiment.joblib")
    artifacts["le_em_ml"] = safe_joblib_load(ML_PATH / "label_encoder_emotion.joblib")
    # DL models + tokenizer + dl encoders
    artifacts["cnn_sent"] = safe_h5_load(DL_PATH / "cnn_sentiment.h5")
    artifacts["lstm_sent"] = safe_h5_load(DL_PATH / "lstm_sentiment.h5")
    artifacts["cnn_em"] = safe_h5_load(DL_PATH / "cnn_emotion.h5")
    artifacts["lstm_em"] = safe_h5_load(DL_PATH / "lstm_emotion.h5")
    artifacts["tokenizer"] = safe_joblib_load(DL_PATH / "tokenizer.joblib")
    artifacts["le_sent_dl"] = safe_joblib_load(DL_PATH / "sentiment_encoder.joblib")
    artifacts["le_em_dl"] = safe_joblib_load(DL_PATH / "emotion_encoder.joblib")
    # ensembles (can be joblib/pickle)
    artifacts["ens_sent"] = safe_joblib_load(ENSEMBLE_PATH / "ensemble_lightgbm_sentiment.pkl")
    artifacts["ens_em"] = safe_joblib_load(ENSEMBLE_PATH / "ensemble_lightgbm_emotion.pkl")
    return artifacts

@st.cache_data
def load_prediction_csvs():
    dfs = {}
    candidates = [
        ML_PATH / "predictions_svm_sentiment.csv",
        ML_PATH / "predictions_randomforest_sentiment.csv",
        ML_PATH / "predictions_naivebayes_sentiment.csv",
        DL_PATH / "predictions_cnn_sentiment.csv",
        DL_PATH / "predictions_lstm_sentiment.csv",
        ML_PATH / "predictions_svm_emotion.csv",
        ML_PATH / "predictions_randomforest_emotion.csv",
        ML_PATH / "predictions_naivebayes_emotion.csv",
        DL_PATH / "predictions_cnn_emotion.csv",
        DL_PATH / "predictions_lstm_emotion.csv",
        ENSEMBLE_PATH / "ensemble_lightgbm_sentiment.csv",
        ENSEMBLE_PATH / "ensemble_lightgbm_emotion.csv",
        RULE_PATH / "rule_based_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                dfs[p.name] = ensure_text_col(pd.read_csv(p))
            except Exception:
                pass
    return dfs

artifacts = load_artifacts()
pred_csvs = load_prediction_csvs()

# -------------------------
# Main page: metrics & visuals (kept minimal)
# -------------------------
st.header("Overview")
sent_ens_df = pred_csvs.get("ensemble_lightgbm_sentiment.csv")
em_ens_df = pred_csvs.get("ensemble_lightgbm_emotion.csv")

c1, c2, c3 = st.columns(3)
with c1:
    if sent_ens_df is not None and {"actual", "ensemble_predicted"}.issubset(sent_ens_df.columns):
        sent_acc = (sent_ens_df["actual"].astype(str) == sent_ens_df["ensemble_predicted"].astype(str)).mean() * 100
        st.metric("Sentiment Ensemble Accuracy", f"{sent_acc:.2f}%")
    else:
        st.metric("Sentiment Ensemble Accuracy", "N/A")
with c2:
    if em_ens_df is not None and {"actual", "ensemble_predicted"}.issubset(em_ens_df.columns):
        em_acc = (em_ens_df["actual"].astype(str) == em_ens_df["ensemble_predicted"].astype(str)).mean() * 100
        st.metric("Emotion Ensemble Accuracy", f"{em_acc:.2f}%")
    else:
        st.metric("Emotion Ensemble Accuracy", "N/A")
with c3:
    sample_count = max((len(df) for df in pred_csvs.values()), default=0)
    st.metric("Samples (max rows)", f"{sample_count}")

st.markdown("---")

# Model accuracies bars (compact)
def compute_accuracy(df):
    if df is None:
        return None
    if "predicted" in df.columns and "actual" in df.columns:
        return (df["predicted"].astype(str) == df["actual"].astype(str)).mean()*100
    if "ensemble_predicted" in df.columns and "actual" in df.columns:
        return (df["ensemble_predicted"].astype(str) == df["actual"].astype(str)).mean()*100
    return None

acc_sent = []
acc_em = []
for name, df in pred_csvs.items():
    ln = name.lower()
    if "sentiment" in ln:
        a = compute_accuracy(df)
        if a is not None:
            acc_sent.append((name.replace(".csv",""), round(a,2)))
    if "emotion" in ln:
        a = compute_accuracy(df)
        if a is not None:
            acc_em.append((name.replace(".csv",""), round(a,2)))

left, right = st.columns(2)
with left:
    st.subheader("Sentiment model accuracies")
    if acc_sent:
        df_acc = pd.DataFrame(acc_sent, columns=["Model","Accuracy"]).sort_values("Accuracy", ascending=False)
        fig = px.bar(df_acc, x="Model", y="Accuracy", text="Accuracy", height=300, template="simple_white")
        fig.update_yaxes(range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment CSVs found.")
with right:
    st.subheader("Emotion model accuracies")
    if acc_em:
        df_acc2 = pd.DataFrame(acc_em, columns=["Model","Accuracy"]).sort_values("Accuracy", ascending=False)
        fig2 = px.bar(df_acc2, x="Model", y="Accuracy", text="Accuracy", height=300, template="simple_white")
        fig2.update_yaxes(range=[0,100])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No emotion CSVs found.")

st.markdown("---")

# Confusion matrices (ensemble)
st.subheader("Confusion Matrices (Ensembles)")
cm1, cm2 = st.columns(2)
with cm1:
    if sent_ens_df is not None and {"actual","ensemble_predicted"}.issubset(sent_ens_df.columns):
        y_true = sent_ens_df["actual"].astype(str)
        y_pred = sent_ens_df["ensemble_predicted"].astype(str)
        labels = sorted(list(set(y_true.tolist()+y_pred.tolist())), key=str)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.info("Sentiment ensemble CSV missing (needs 'actual' & 'ensemble_predicted').")
with cm2:
    if em_ens_df is not None and {"actual","ensemble_predicted"}.issubset(em_ens_df.columns):
        y_true = em_ens_df["actual"].astype(str)
        y_pred = em_ens_df["ensemble_predicted"].astype(str)
        labels = sorted(list(set(y_true.tolist()+y_pred.tolist())), key=str)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig2, ax2 = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
        st.pyplot(fig2)
    else:
        st.info("Emotion ensemble CSV missing (needs 'actual' & 'ensemble_predicted').")

st.markdown("---")

# Single word cloud (positive comments)
st.subheader("Word Cloud — Positive Comments")
rb = pred_csvs.get("rule_based_predictions.csv")
if rb is not None and "text" in rb.columns and "sentiment" in rb.columns:
    pos_texts = rb[rb["sentiment"].astype(str).str.contains("1")]["text"].dropna().astype(str).tolist()
    if pos_texts:
        wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(pos_texts))
        st.image(wc.to_array(), use_column_width=True)
    else:
        st.info("No positive comments found.")
else:
    st.info("rule_based_predictions.csv not available (needed for word cloud).")


# -------------------------
# Sidebar: Real-time prediction (per-model + ensemble) — display results in sidebar
# -------------------------
st.sidebar.header("Real-time prediction (per-model + ensemble)")
task = st.sidebar.selectbox("Task", ["sentiment", "emotion"])
input_text = st.sidebar.text_area("Enter text to analyze", height=140)
if st.sidebar.button("Predict"):
    if not input_text or not input_text.strip():
        st.sidebar.warning("Enter some text first.")
    else:
        # prepare features
        tfidf = artifacts.get("tfidf") if (artifacts := load_artifacts()) else None
        tokenizer = artifacts.get("tokenizer") if artifacts else None
        text_ml = preprocess_ml(input_text)
        Xvec = None
        if tfidf is not None:
            try:
                Xvec = tfidf.transform([text_ml])
            except Exception:
                Xvec = None

        # Gather per-model outputs
        per_model_rows = []
        if task == "sentiment":
            ml_list = [("rf_sent","RandomForest","le_sent_ml"), ("svm_sent","SVM","le_sent_ml"), ("nb_sent","NaiveBayes","le_sent_ml")]
            dl_list = [("cnn_sent","CNN","le_sent_dl"), ("lstm_sent","LSTM","le_sent_dl")]
        else:
            ml_list = [("rf_em","RandomForest","le_em_ml"), ("svm_em","SVM","le_em_ml"), ("nb_em","NaiveBayes","le_em_ml")]
            dl_list = [("cnn_em","CNN","le_em_dl"), ("lstm_em","LSTM","le_em_dl")]

        # ML predictions
        for key, label_name, le_key in ml_list:
            model = artifacts.get(key)
            le_ml = artifacts.get(le_key)
            if model is not None and Xvec is not None:
                try:
                    raw = model.predict(Xvec)[0]
                    pred_label = decode_prediction(raw, le_ml)
                except Exception:
                    pred_label = "err"
            else:
                pred_label = "n/a"
            per_model_rows.append((label_name, pred_label))

        # DL predictions
        seq = None
        if tokenizer is not None:
            try:
                seq = tokenizer.texts_to_sequences([input_text]); seq = pad_sequences(seq, maxlen=100)
            except Exception:
                seq = None

        for key, label_name, le_key in dl_list:
            mdl = artifacts.get(key)
            le_dl = artifacts.get(le_key)
            if mdl is not None and seq is not None:
                try:
                    proba = mdl.predict(seq, verbose=0)
                    idx = int(np.argmax(proba, axis=1)[0])
                    pred_label = decode_prediction(idx, le_dl)
                except Exception:
                    pred_label = "err"
            else:
                pred_label = "n/a"
            per_model_rows.append((label_name, pred_label))

        # rule-based (only for sentiment)
        if task == "sentiment":
            per_model_rows.append(("VADER", vader_label(input_text)))
            per_model_rows.append(("TextBlob", textblob_label(input_text)))

        # show per-model predictions in sidebar
        st.sidebar.subheader("Base model predictions")
        st.sidebar.table(pd.DataFrame(per_model_rows, columns=["Model","Prediction"]))

        # Ensemble predict: construct feature vector in expected order
        ens_obj = artifacts.get("ens_sent") if task=="sentiment" else artifacts.get("ens_em")
        if ens_obj is None:
            st.sidebar.error("Ensemble model is not available for this task.")
        else:
            # unpack ensemble object
            model_obj, le_obj, feat_cols = None, None, None
            if isinstance(ens_obj, (list,tuple)):
                model_obj = ens_obj[0]; le_obj = ens_obj[1] if len(ens_obj)>1 else None; feat_cols = ens_obj[2] if len(ens_obj)>2 else None
            elif isinstance(ens_obj, dict):
                model_obj = ens_obj.get("model"); le_obj = ens_obj.get("label_encoder"); feat_cols = ens_obj.get("feature_columns")
            else:
                model_obj = ens_obj

            if not feat_cols:
                feat_cols = ['rf','svm','nb','cnn','lstm','vader','textblob'] if task=="sentiment" else ['rf','svm','nb','cnn','lstm']

            # build a dict mapping short names to predictions
            pred_map = {m.lower(): p for m,p in per_model_rows}
            feat_values = [str(pred_map.get(c.lower(), "n/a")) for c in feat_cols]

            try:
                if le_obj is not None:
                    feat_enc = [int(le_obj.transform([v])[0]) for v in feat_values]
                    proba = None
                    if hasattr(model_obj, "predict_proba"):
                        proba = model_obj.predict_proba([feat_enc])[0]
                    elif hasattr(model_obj, "booster_"):
                        proba = model_obj.predict_proba([feat_enc])[0]
                    if proba is not None:
                        arg = int(np.argmax(proba))
                        final_label = le_obj.inverse_transform([arg])[0]
                        confidence = float(proba[arg])
                    else:
                        pred_enc = model_obj.predict([feat_enc])[0]
                        final_label = le_obj.inverse_transform([int(pred_enc)])[0]
                        confidence = 1.0
                else:
                    pred = model_obj.predict([feat_values])[0]
                    final_label = str(pred)
                    confidence = 1.0

                st.sidebar.success(f"Ensemble: {final_label}  ({confidence*100:.1f}%)")
            except Exception:
                st.sidebar.error("Ensemble prediction failed (encoding/model mismatch).")

st.markdown("---")

# -------------------------
# Download combined ensemble CSVs at end
# -------------------------
st.header("Download — combined ensemble outputs")
sent_csv = pred_csvs.get("ensemble_lightgbm_sentiment.csv")
em_csv = pred_csvs.get("ensemble_lightgbm_emotion.csv")
combined = None
if sent_csv is not None:
    sent_csv = sent_csv.rename(columns=lambda c: c.strip())
if em_csv is not None:
    em_csv = em_csv.rename(columns=lambda c: c.strip())

if sent_csv is not None and em_csv is not None:
    s = sent_csv.copy(); s["task"] = "sentiment"
    e = em_csv.copy(); e["task"] = "emotion"
    combined = pd.concat([s, e], ignore_index=True)
elif sent_csv is not None:
    combined = sent_csv.copy(); combined["task"] = "sentiment"
elif em_csv is not None:
    combined = em_csv.copy(); combined["task"] = "emotion"

if combined is not None:
    st.write(f"Rows: {len(combined)}")
    st.dataframe(combined.head(40))
    st.download_button("Download combined ensemble CSV", combined.to_csv(index=False).encode("utf-8"), "ensemble_combined.csv")
else:
    st.info("No ensemble CSVs found. Run ensemble script to generate them.")

st.caption("Minimal dashboard — real-time predictions shown in sidebar.")
