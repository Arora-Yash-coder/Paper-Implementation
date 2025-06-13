import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# ========== TRAIN & SAVE FUNCTION ========== #
def train_evaluate_save(model, model_name, task_name, X_train, X_test, y_train, y_test, label_encoder, raw_texts_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # Decode for human-readable output
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    class_names = [str(c) for c in label_encoder.classes_]

    # Save classification report
    report = classification_report(
        y_test_decoded,
        y_pred_decoded,
        labels=class_names,
        target_names=class_names,
        zero_division=0
    )

    print(f"{model_name} - {task_name.lower()} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(report)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_{task_name.lower()}.joblib")
    joblib.dump(model, model_path)

    # Save predictions
    pred_df = pd.DataFrame({
        'text': raw_texts_test,
        'actual': y_test_decoded,
        'predicted': y_pred_decoded
    })
    pred_file = f"predictions_{task_name.lower()}.csv"
    pred_df.to_csv(os.path.join(OUTPUT_DIR, pred_file), index=False)

# ========== MAIN ========== #
def main():
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=['comment', 'sentiment', 'emotion'], inplace=True)
    df['clean_text'] = df['comment'].astype(str).apply(preprocess)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])

    # Save the vectorizer
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))

    # Label encoders
    le_sentiment = LabelEncoder()
    y_sentiment = le_sentiment.fit_transform(df['sentiment'])
    joblib.dump(le_sentiment, os.path.join(OUTPUT_DIR, "label_encoder_sentiment.joblib"))

    le_emotion = LabelEncoder()
    y_emotion = le_emotion.fit_transform(df['emotion'])
    joblib.dump(le_emotion, os.path.join(OUTPUT_DIR, "label_encoder_emotion.joblib"))

    models = [
        (MultinomialNB(), 'NaiveBayes'),
        (SVC(kernel='linear', probability=True), 'SVM'),
        (RandomForestClassifier(n_estimators=100, random_state=42), 'RandomForest')
    ]

    # --- Sentiment Classification --- #
    print("\n--- Sentiment Classification ---")
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y_sentiment, df['comment'].values, test_size=0.2, random_state=42
    )

    for model, name in models:
        train_evaluate_save(
            model, name, "Sentiment", X_train, X_test,
            y_train, y_test, le_sentiment, texts_test
        )

    # --- Emotion Classification --- #
    print("\n--- Emotion Classification ---")
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y_emotion, df['comment'].values, test_size=0.2, random_state=42
    )

    for model, name in models:
        train_evaluate_save(
            model, name, "Emotion", X_train, X_test,
            y_train, y_test, le_emotion, texts_test
        )

if __name__ == "__main__":
    main()
