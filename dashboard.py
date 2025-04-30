import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Load models and utilities
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_vectorizer():
    return joblib.load("models/tfidf_vectorizer.joblib")

@st.cache_resource
def load_ml_models():
    return {
        "Naive Bayes": joblib.load("models/naive_bayes_model.joblib"),
        "SVM": joblib.load("models/svm_model.joblib"),
        "Random Forest": joblib.load("models/random_forest_model.joblib")
    }

@st.cache_resource
def load_tokenizer():
    return joblib.load("models/tokenizer_dl.pkl")

@st.cache_resource
def load_dl_models():
    return {
        "CNN": load_model("models/cnn_model.h5"),
        "LSTM": load_model("models/lstm_model.h5")
    }

@st.cache_data
def load_all_data():
    rule_based_df = pd.read_csv("hmpv_rule_based_results.csv")
    ml_df = pd.read_csv("hmpv_ml_results.csv")
    dl_df = pd.read_csv("hmpv_dl_results.csv")
    return rule_based_df, ml_df, dl_df

# Text preprocessing
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', text.lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

# Inference functions
def predict_rule_based(text, method):
    if method == "VADER":
        analyzer = load_vader()
        score = analyzer.polarity_scores(text)["compound"]
        return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
    else:
        score = TextBlob(text).sentiment.polarity
        return "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"

def predict_ml(text, model_name):
    vectorizer = load_vectorizer()
    model = load_ml_models()[model_name]
    vec = vectorizer.transform([text])
    label = model.predict(vec)[0]
    return ["Negative", "Neutral", "Positive"][label + 1]

def predict_dl(text, model_name):
    tokenizer = load_tokenizer()
    model = load_dl_models()[model_name]
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded)
    label = np.argmax(pred, axis=1)[0]
    return ["Negative", "Neutral", "Positive"][label]

# Sidebar
st.sidebar.title("📍 Realtime Sentiment Prediction")
user_input = st.sidebar.text_area("Enter a comment:")
model_choice = st.sidebar.selectbox("Choose sentiment model", [
    "Rule-based (VADER vs TextBlob)",
    "ML-based (Naive Bayes, SVM, Random Forest)",
    "DL-based (CNN, LSTM)"
])

sub_model = None
if model_choice == "Rule-based (VADER vs TextBlob)":
    sub_model = st.sidebar.radio("Select rule-based model", ["VADER", "TextBlob"])
elif model_choice == "ML-based (Naive Bayes, SVM, Random Forest)":
    sub_model = st.sidebar.radio("Select ML model", ["Naive Bayes", "SVM", "Random Forest"])
elif model_choice == "DL-based (CNN, LSTM)":
    sub_model = st.sidebar.radio("Select DL model", ["CNN", "LSTM"])

if st.sidebar.button("Analyze"):
    cleaned = clean_text(user_input)
    if model_choice == "Rule-based (VADER vs TextBlob)":
        sentiment = predict_rule_based(cleaned, sub_model)
    elif model_choice == "ML-based (Naive Bayes, SVM, Random Forest)":
        sentiment = predict_ml(cleaned, sub_model)
    elif model_choice == "DL-based (CNN, LSTM)":
        sentiment = predict_dl(cleaned, sub_model)
    st.sidebar.success(f"Predicted Sentiment: {sentiment}")

# Title and loading data
st.title("📊 Sentiment Analysis Dashboard")
progress_bar = st.progress(0)
rule_based_df, ml_df, dl_df = load_all_data()
progress_bar.progress(100)

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.subheader("Rule-based Sentiment Analysis Data")
    st.write(rule_based_df[['comment', 'vader_sentiment', 'textblob_sentiment']])

    st.subheader("ML-based Sentiment Analysis Data")
    st.write(ml_df[['comment', 'sentiment', 'naive_bayes_prediction', 'svm_prediction', 'random_forest_prediction']])

    st.subheader("DL-based Sentiment Analysis Data")
    st.write(dl_df[['comment', 'cnn_sentiment', 'lstm_sentiment']])

# Plotting
st.subheader("Sentiment Analysis Model Performance Comparison")

# Rule-based
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x='vader_sentiment', data=rule_based_df, ax=ax[0], palette='Blues')
ax[0].set_title('VADER Sentiment Distribution')
sns.countplot(x='textblob_sentiment', data=rule_based_df, ax=ax[1], palette='Greens')
ax[1].set_title('TextBlob Sentiment Distribution')
st.pyplot(fig)

# ML-based
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.countplot(x='sentiment', data=ml_df, ax=ax[0], palette='Purples')
ax[0].set_title('Overall ML Sentiment Distribution')
sns.countplot(x='naive_bayes_prediction', data=ml_df, ax=ax[1], palette='Oranges')
ax[1].set_title('Naive Bayes Sentiment Distribution')
sns.countplot(x='svm_prediction', data=ml_df, ax=ax[2], palette='Reds')
ax[2].set_title('SVM Sentiment Distribution')
st.pyplot(fig)

# DL-based
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x='cnn_sentiment', data=dl_df, ax=ax[0], palette='Purples')
ax[0].set_title('CNN Sentiment Distribution')
sns.countplot(x='lstm_sentiment', data=dl_df, ax=ax[1], palette='Blues')
ax[1].set_title('LSTM Sentiment Distribution')
st.pyplot(fig)
