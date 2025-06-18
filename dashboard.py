import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Advanced Sentiment & Emotion Dashboard", layout="wide")

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
    def load_csv(path):
        return pd.read_csv(path)

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

    return {"rule_based": rule_based, "ml": ml_preds, "dl": dl_preds}

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', str(text).lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

def sidebar_predictions(resources):
    st.sidebar.title("🔮 Real-time Analysis")
    user_input = st.sidebar.text_area("Enter text:")
    analysis_type = st.sidebar.radio("Analysis Type", ["Sentiment", "Emotion"])
    model_type = st.sidebar.selectbox("Model Type", ["Rule-based", "ML", "DL"]) if analysis_type == "Sentiment" else st.sidebar.selectbox("Model Type", ["ML", "DL"])
    model_choice = st.sidebar.radio("Choose Model", ["VADER", "TextBlob"]) if model_type == "Rule-based" else st.sidebar.radio("Choose Model", ["naive_bayes", "svm", "random_forest"] if model_type == "ML" else ["cnn", "lstm"])

    if st.sidebar.button("Analyze"):
        if not user_input:
            st.sidebar.warning("Please enter some text")
            return
        cleaned = clean_text(user_input)
        try:
            if model_type == "Rule-based":
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
                    result = ["Negative", "Neutral", "Positive"][pred + 1]
                else:
                    pred = resources['ml_emotion'][model_choice].predict(vec)[0]
                    result = resources['emotion_encoder'].inverse_transform([pred])[0].lower()
            else:
                seq = resources['tokenizer'].texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=100, padding='post')
                if analysis_type == "Sentiment":
                    pred = resources['dl_sentiment'][model_choice].predict(padded)
                    result = ["Negative", "Neutral", "Positive"][np.argmax(pred)]
                else:
                    pred = resources['dl_emotion'][model_choice].predict(padded)
                    result = resources['emotion_encoder'].inverse_transform([np.argmax(pred)])[0].lower()
            st.sidebar.success(f"**{analysis_type} Result**: {result}")
        except Exception as e:
            st.sidebar.error(f"Analysis failed: {str(e)}")

def create_sentiment_plot(data, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x=pd.Categorical(data, categories=SENTIMENT_VALUES), ax=ax, order=SENTIMENT_VALUES, palette='viridis')
    ax.set_title(title)
    return fig

def create_emotion_plot(data, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(y=pd.Categorical(data, categories=EMOTION_CATEGORIES), ax=ax, order=EMOTION_CATEGORIES, palette='Set2')
    ax.set_title(title)
    return fig

def generate_wordcloud(texts):
    all_text = ' '.join(texts.astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def main():
    resources = load_resources()
    data = load_all_data()

    st.title("📊 Advanced Sentiment & Emotion Dashboard")

    with st.expander("📝 Show Raw Data"):
        st.dataframe(data['rule_based'])

    st.header("📈 Sentiment Analysis")
    st.subheader("Rule-based Models")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(create_sentiment_plot(data['rule_based']['vader_sentiment'], "VADER Sentiment"))
    with col2:
        st.pyplot(create_sentiment_plot(data['rule_based']['textblob_sentiment'], "TextBlob Sentiment"))

    st.subheader("ML Models")
    for model in ["naive_bayes_sentiment", "svm_sentiment", "random_forest_sentiment"]:
        st.pyplot(create_sentiment_plot(data['ml'][model]['predicted'], f"{model.replace('_', ' ').title()}"))

    st.subheader("DL Models")
    for model in ["cnn_sentiment", "lstm_sentiment"]:
        st.pyplot(create_sentiment_plot(data['dl'][model]['predicted'], f"{model.replace('_', ' ').title()}"))

    st.header("🎭 Emotion Analysis")
    st.subheader("ML Models")
    for model in ["naive_bayes_emotion", "svm_emotion", "random_forest_emotion"]:
        st.pyplot(create_emotion_plot(data['ml'][model]['predicted'], f"{model.replace('_', ' ').title()}"))

    st.subheader("DL Models")
    for model in ["cnn_emotion", "lstm_emotion"]:
        st.pyplot(create_emotion_plot(data['dl'][model]['predicted'], f"{model.replace('_', ' ').title()}"))

    st.header("☁️ Word Cloud (Rule-based Comments)")
    generate_wordcloud(data['rule_based']['comment'])

    

if __name__ == "__main__":
    sidebar_predictions(load_resources())
    main()
