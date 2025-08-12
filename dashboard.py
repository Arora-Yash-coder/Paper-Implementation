import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob

st.set_page_config(page_title="HMPV Sentiment & Emotion Dashboard", layout="wide")

SENTIMENT_VALUES = [-1, 0, 1]
EMOTION_CATEGORIES = [
    'happy', 'joy', 'fear', 'anger', 'enthusiasm',
    'sad', 'relief', 'sympathy', 'surprise', 'disgust', 'unemotional'
]

@st.cache_resource
def load_resources():
    return {
        'vader': SentimentIntensityAnalyzer(),
        'tfidf': joblib.load("results/ml_models/tfidf_vectorizer.joblib"),
        'ml_sentiment': {
            'naive_bayes': joblib.load("results/ml_models/naivebayes_sentiment.joblib"),
            'svm': joblib.load("results/ml_models/svm_sentiment.joblib"),
            'random_forest': joblib.load("results/ml_models/randomforest_sentiment.joblib"),
        },
        'ml_emotion': {
            'naive_bayes': joblib.load("results/ml_models/naivebayes_emotion.joblib"),
            'svm': joblib.load("results/ml_models/svm_emotion.joblib"),
            'random_forest': joblib.load("results/ml_models/randomforest_emotion.joblib"),
        },
        'tokenizer': joblib.load("results/dl_models/tokenizer.joblib"),
        'dl_sentiment': {
            'cnn': load_model("results/dl_models/cnn_sentiment.h5"),
            'lstm': load_model("results/dl_models/lstm_sentiment.h5"),
        },
        'dl_emotion': {
            'cnn': load_model("results/dl_models/cnn_emotion.h5"),
            'lstm': load_model("results/dl_models/lstm_emotion.h5"),
        },
        'sentiment_encoder': joblib.load("results/dl_models/sentiment_encoder.joblib"),
        'emotion_encoder': joblib.load("results/dl_models/emotion_encoder.joblib")
    }

@st.cache_data
def load_all_data():
    def load_csv(path): return pd.read_csv(path)

    rule_based = load_csv("results/rule_based_models/rule_based_predictions.csv")
    ml_preds = {
        "naive_bayes_sentiment": load_csv("results/ml_models/predictions_naivebayes_sentiment.csv"),
        "svm_sentiment": load_csv("results/ml_models/predictions_svm_sentiment.csv"),
        "random_forest_sentiment": load_csv("results/ml_models/predictions_randomforest_sentiment.csv"),
        "naive_bayes_emotion": load_csv("results/ml_models/predictions_naivebayes_emotion.csv"),
        "svm_emotion": load_csv("results/ml_models/predictions_svm_emotion.csv"),
        "random_forest_emotion": load_csv("results/ml_models/predictions_randomforest_emotion.csv"),
    }
    dl_preds = {
        "cnn_sentiment": load_csv("results/dl_models/predictions_cnn_sentiment.csv"),
        "lstm_sentiment": load_csv("results/dl_models/predictions_lstm_sentiment.csv"),
        "cnn_emotion": load_csv("results/dl_models/predictions_cnn_emotion.csv"),
        "lstm_emotion": load_csv("results/dl_models/predictions_lstm_emotion.csv"),
    }
    ensemble = {
        "ensemble_sentiment": load_csv("results/ensemble/ensemble_sentiment_predictions.csv"),
        "ensemble_emotion": load_csv("results/ensemble/ensemble_emotion_predictions.csv"),
    }

    return {"rule_based": rule_based, "ml": ml_preds, "dl": dl_preds, "ensemble": ensemble}

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', str(text).lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

def display_sidebar():
    st.sidebar.title("📊 Dashboard Navigation")
    st.sidebar.markdown("Use the options below to explore predictions and model evaluations.")
    section = st.sidebar.radio("Choose Section", ["Project Overview", "Sentiment Analysis", "Emotion Analysis", "WordCloud"])
    return section
