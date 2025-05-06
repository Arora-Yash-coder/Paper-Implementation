import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import numpy as np
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Advanced Sentiment & Emotion Dashboard", layout="wide")

# Constants
EMOTION_CATEGORIES = [
    'happy', 'joy', 'fear', 'anger', 'enthusiasm',
    'sad', 'relief', 'sympathy', 'surprise', 'disgust', 'unemotional'
]
SENTIMENT_VALUES = [-1, 0, 1]

# Load models and utilities
@st.cache_resource
def load_resources():
    return {
        'vader': joblib.load("models/vader_analyzer.joblib"),
        'textblob': joblib.load("models/textblob_analyzer.joblib"),
        'tfidf': joblib.load("models/tfidf_vectorizer.joblib"),
        'ml_sentiment': {
            'naive_bayes': joblib.load("models/naive_bayes_sentiment.joblib"),
            'svm': joblib.load("models/svm_sentiment.joblib"),
            'random_forest': joblib.load("models/random_forest_sentiment.joblib")
        },
        'ml_emotion': {
            'naive_bayes': joblib.load("models/naive_bayes_emotion.joblib"),
            'svm': joblib.load("models/svm_emotion.joblib"),
            'random_forest': joblib.load("models/random_forest_emotion.joblib")
        },
        'tokenizer': joblib.load("models/dl_tokenizer.joblib"),
        'dl_sentiment': {
            'cnn': load_model("models/cnn_sentiment.h5"),
            'lstm': load_model("models/lstm_sentiment.h5")
        },
        'dl_emotion': {
            'cnn': load_model("models/cnn_emotion.h5"),
            'lstm': load_model("models/lstm_emotion.h5")
        },
        'emotion_encoder': joblib.load("models/emotion_encoder.joblib")
    }

@st.cache_data
def load_all_data():
    # Normalize emotion columns to lowercase string
    def normalize_emotion(df):
        for col in df.columns:
            if 'emotion' in col.lower():
                df[col] = df[col].astype(str).str.lower()
        return df
    return {
        'rule_based': normalize_emotion(pd.read_csv("results/hmpv_sentiment_results.csv")),
        'ml': normalize_emotion(pd.read_csv("results/hmpv_ml_results.csv")),
        'dl': normalize_emotion(pd.read_csv("results/hmpv_dl_results.csv"))
    }

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', str(text).lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

def sidebar_predictions(resources):
    st.sidebar.title("🔮 Real-time Analysis")
    user_input = st.sidebar.text_area("Enter text:")
    analysis_type = st.sidebar.radio("Analysis Type", ["Sentiment", "Emotion"])
    if analysis_type == "Sentiment":
        model_type = st.sidebar.selectbox("Model Type", ["Rule-based", "ML", "DL"])
    else:
        model_type = st.sidebar.selectbox("Model Type", ["ML", "DL"])
    model_choice = None
    if model_type == "Rule-based":
        model_choice = st.sidebar.radio("Choose Model", ["VADER", "TextBlob"])
    elif model_type == "ML":
        model_choice = st.sidebar.radio("Choose Model", ["naive_bayes", "svm", "random_forest"])
    else:
        model_choice = st.sidebar.radio("Choose Model", ["cnn", "lstm"])
    if st.sidebar.button("Analyze"):
        if not user_input:
            st.sidebar.warning("Please enter some text to analyze")
            return
        cleaned = clean_text(user_input)
        result = None
        try:
            if model_type == "Rule-based":
                if analysis_type == "Sentiment":
                    if model_choice == "VADER":
                        score = resources['vader'].polarity_scores(cleaned)['compound']
                        result = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                    else:
                        score = TextBlob(cleaned).sentiment.polarity
                        result = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
            elif model_type == "ML":
                vec = resources['tfidf'].transform([cleaned])
                if analysis_type == "Sentiment":
                    pred = resources['ml_sentiment'][model_choice].predict(vec)[0]
                    result = ["Negative", "Neutral", "Positive"][pred+1]
                else:
                    pred = resources['ml_emotion'][model_choice].predict(vec)[0]
                    result = resources['emotion_encoder'].inverse_transform([pred])[0].lower()
            else:
                seq = resources['tokenizer'].texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=100, padding='post')
                if analysis_type == "Sentiment":
                    pred = resources['dl_sentiment'][model_choice].predict(padded)
                    pred_idx = np.argmax(pred, axis=1)[0]
                    result = ["Negative", "Neutral", "Positive"][pred_idx]
                else:
                    pred = resources['dl_emotion'][model_choice].predict(padded)
                    pred_idx = np.argmax(pred, axis=1)[0]
                    result = resources['emotion_encoder'].inverse_transform([pred_idx])[0].lower()
            st.sidebar.success(f"**{analysis_type} Result**: {result}")
        except Exception as e:
            st.sidebar.error(f"Analysis failed: {str(e)}")

def create_sentiment_plots(data, title):
    # Ensure all sentiment values are present as categories
    data = pd.Categorical(data, categories=SENTIMENT_VALUES)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x=data, ax=ax, palette='viridis', order=SENTIMENT_VALUES)
    ax.set_title(title)
    ax.set_xlabel('Sentiment (-1=Negative, 0=Neutral, 1=Positive)')
    return fig

def create_emotion_plots(data, title):
    # Ensure all emotion categories are present as categories
    data = pd.Categorical(data, categories=EMOTION_CATEGORIES)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(y=data, ax=ax, palette='Set2', order=EMOTION_CATEGORIES)
    ax.set_title(title)
    return fig

def main():
    try:
        resources = load_resources()
        data = load_all_data()
        st.title("📊 Advanced Sentiment & Emotion Analysis Dashboard")
        with st.expander("Show Raw Data"):
            tab1, tab2, tab3 = st.tabs(["Rule-based", "ML", "DL"])
            with tab1:
                st.dataframe(data['rule_based'][['comment', 'vader_sentiment', 'textblob_sentiment']])
            with tab2:
                st.dataframe(data['ml'][['comment', 'sentiment', 'naive_bayes_sentiment', 
                                      'svm_sentiment', 'random_forest_sentiment', 
                                      'naive_bayes_emotion', 'svm_emotion', 'random_forest_emotion']])
            with tab3:
                st.dataframe(data['dl'][['comment', 'cnn_sentiment', 'lstm_sentiment',
                                      'cnn_emotion', 'lstm_emotion']])

        # Sentiment Analysis Section
        st.header("📈 Sentiment Analysis")
        st.markdown("#### Rule-based Models")
        col1, col2 = st.columns(2)
        with col1:
            fig = create_sentiment_plots(data['rule_based']['vader_sentiment'], "VADER Sentiment")
            st.pyplot(fig)
        with col2:
            fig = create_sentiment_plots(data['rule_based']['textblob_sentiment'], "TextBlob Sentiment")
            st.pyplot(fig)

        st.markdown("#### ML Models")
        col3, col4, col5 = st.columns(3)
        with col3:
            fig = create_sentiment_plots(data['ml']['naive_bayes_sentiment'], "Naive Bayes Sentiment")
            st.pyplot(fig)
        with col4:
            fig = create_sentiment_plots(data['ml']['svm_sentiment'], "SVM Sentiment")
            st.pyplot(fig)
        with col5:
            fig = create_sentiment_plots(data['ml']['random_forest_sentiment'], "Random Forest Sentiment")
            st.pyplot(fig)

        st.markdown("#### DL Models")
        # Ensure DL sentiment columns are categorical with all possible values
        for col in ['cnn_sentiment', 'lstm_sentiment']:
            data['dl'][col] = pd.Categorical(data['dl'][col], categories=SENTIMENT_VALUES)
        col6, col7 = st.columns(2)
        with col6:
            fig = create_sentiment_plots(data['dl']['cnn_sentiment'], "CNN Sentiment")
            st.pyplot(fig)
        with col7:
            fig = create_sentiment_plots(data['dl']['lstm_sentiment'], "LSTM Sentiment")
            st.pyplot(fig)

        # Emotion Analysis Section
        st.header("🎭 Emotion Analysis")
        st.markdown("#### ML Models")
        col8, col9, col10 = st.columns(3)
        with col8:
            fig = create_emotion_plots(data['ml']['naive_bayes_emotion'], "Naive Bayes Emotions")
            st.pyplot(fig)
        with col9:
            fig = create_emotion_plots(data['ml']['svm_emotion'], "SVM Emotions")
            st.pyplot(fig)
        with col10:
            fig = create_emotion_plots(data['ml']['random_forest_emotion'], "Random Forest Emotions")
            st.pyplot(fig)

        st.markdown("#### DL Models")
        # Ensure DL emotion columns are categorical with all possible categories
        for col in ['cnn_emotion', 'lstm_emotion']:
            data['dl'][col] = pd.Categorical(data['dl'][col], categories=EMOTION_CATEGORIES)
        col11, col12 = st.columns(2)
        with col11:
            fig = create_emotion_plots(data['dl']['cnn_emotion'], "CNN Emotions")
            st.pyplot(fig)
        with col12:
            fig = create_emotion_plots(data['dl']['lstm_emotion'], "LSTM Emotions")
            st.pyplot(fig)

        # Model Comparison Section
        st.header("🔍 Model Performance Comparison")
        try:
            report_files = [f for f in os.listdir("results") if f.endswith("_report.csv")]
            model_reports = {}
            for file in report_files:
                model_name = file.replace("_report.csv", "").replace("_", " ").title()
                df = pd.read_csv(f"results/{file}", index_col=0)
                model_reports[model_name] = df
            if model_reports:
                selected_report = st.selectbox("Select Classification Report", model_reports.keys())
                st.dataframe(model_reports[selected_report])
                # F1 Score Comparison
                f1_scores = {}
                for model, report in model_reports.items():
                    try:
                        possible_keys = ['weighted avg', 'weighted_avg', 'Weighted Avg', 'macro avg']
                        for key in possible_keys:
                            if key in report.index:
                                f1_scores[model] = report.loc[key, 'f1-score']
                                break
                    except Exception as e:
                        st.warning(f"Couldn't extract F1 score for {model}: {str(e)}")
                if f1_scores:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette='rocket')
                    plt.title('F1 Score Comparison Across Models')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            else:
                st.warning("No classification reports found in results directory")
        except Exception as e:
            st.error(f"Error loading reports: {str(e)}")

    except Exception as e:
        st.error(f"Dashboard initialization failed: {str(e)}")

if __name__ == "__main__":
    sidebar_predictions(load_resources())
    main()
