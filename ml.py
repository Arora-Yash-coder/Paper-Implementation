import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# ========== CONFIG ========== #
INPUT_FILE = 'input_data.csv'
OUTPUT_DIR = 'results/ml_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== TEXT CLEANING ========== #
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

def train_predict_save(model, model_name, task_name, X, y, label_encoder, raw_texts):
    model.fit(X, y)
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)

    y_decoded = label_encoder.inverse_transform(y)
    y_pred_decoded = label_encoder.inverse_transform(predictions)

    report = classification_report(y_decoded, y_pred_decoded, zero_division=0)
    print(f"{model_name} - {task_name} - Accuracy: {acc:.4f}")
    print(report)

    # Save model
    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{model_name.lower()}_{task_name.lower()}.joblib"))

    # Save predictions
    pred_df = pd.DataFrame({
        'text': raw_texts,
        'actual': y_decoded,
        'predicted': y_pred_decoded
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"predictions_{model_name.lower()}_{task_name.lower()}.csv"), index=False)

def main():
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=['comment', 'sentiment', 'emotion'], inplace=True)
    df['clean_text'] = df['comment'].astype(str).apply(preprocess)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))

    # Label encode sentiment
    le_sentiment = LabelEncoder()
    y_sentiment = le_sentiment.fit_transform(df['sentiment'])
    joblib.dump(le_sentiment, os.path.join(OUTPUT_DIR, "label_encoder_sentiment.joblib"))

    # Label encode emotion
    le_emotion = LabelEncoder()
    y_emotion = le_emotion.fit_transform(df['emotion'])
    joblib.dump(le_emotion, os.path.join(OUTPUT_DIR, "label_encoder_emotion.joblib"))

    models = [
        (MultinomialNB(), 'NaiveBayes'),
        (SVC(kernel='linear', probability=True), 'SVM'),
        (RandomForestClassifier(n_estimators=100, random_state=42), 'RandomForest')
    ]

    print("\n--- Sentiment Classification (Full Dataset) ---")
    for model, name in models:
        train_predict_save(model, name, "Sentiment", X, y_sentiment, le_sentiment, df['comment'].values)

    print("\n--- Emotion Classification (Full Dataset) ---")
    for model, name in models:
        train_predict_save(model, name, "Emotion", X, y_emotion, le_emotion, df['comment'].values)

if __name__ == "__main__":
    main()
