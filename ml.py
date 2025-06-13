# ml_analysis.py
import pandas as pd
import re
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Configuration
DATA_PATH = "data/hmpv_comments_labeled.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
EMOTION_CATEGORIES = [
    'happy', 'joy', 'fear', 'anger', 'enthusiasm',
    'sad', 'relief', 'sympathy', 'surprise', 'disgust', 'unemotional'
]

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', str(text).lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

def validate_labels(df):
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    df['sentiment'] = df['sentiment'].fillna(0).astype(int)
    df = df[df['sentiment'].isin([-1, 0, 1])]
    df['emotion'] = df['emotion'].apply(lambda x: x if x in EMOTION_CATEGORIES else 'unemotional')
    return df

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['clean_comment'] = df['comment'].apply(clean_text)
    df = validate_labels(df)

    le = LabelEncoder()
    le.fit(EMOTION_CATEGORIES)
    df['emotion_encoded'] = le.transform(df['emotion'])

    print("Unique emotions in labeled data:", df['emotion'].unique())
    print("Emotion distribution:\n", df['emotion'].value_counts())

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(df['clean_comment'])
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(le, f"{MODELS_DIR}/emotion_encoder.joblib")

    models = {
        "naive_bayes": MultinomialNB(),
        "svm": SVC(kernel='linear', probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    # ===== Sentiment Analysis =====
    print("\n🔍 Training Sentiment Models...")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42
    )
    for name, model in models.items():
        print(f"🚀 Training {name} for sentiment...")
        model.fit(X_train_s, y_train_s)
        df[f'{name}_sentiment'] = model.predict(X)
        report = classification_report(
            y_test_s, model.predict(X_test_s),
            target_names=['Negative', 'Neutral', 'Positive'],
            output_dict=True
        )
        pd.DataFrame(report).transpose().to_csv(f"{RESULTS_DIR}/{name}_sentiment_report.csv")
        joblib.dump(model, f"{MODELS_DIR}/{name}_sentiment.joblib")

    # ===== Emotion Analysis =====
    print("\n🎭 Training Emotion Models...")
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X, df['emotion_encoded'], test_size=0.2, stratify=df['emotion_encoded'], random_state=42
    )
    unique_classes = np.unique(y_train_e)
    emotion_cols = []
    if len(unique_classes) < 2:
        print(f"\n⚠️ Skipping emotion analysis - Only 1 class ({le.inverse_transform(unique_classes)[0]}) detected")
    else:
        for name, model in models.items():
            print(f"🚀 Training {name} for emotion...")
            model.fit(X_train_e, y_train_e)
            df[f'{name}_emotion_encoded'] = model.predict(X)
            df[f'{name}_emotion'] = le.inverse_transform(df[f'{name}_emotion_encoded'])
            emotion_cols.extend([f'{name}_emotion', f'{name}_emotion_encoded'])
            report = classification_report(
                y_test_e, model.predict(X_test_e),
                labels=range(len(EMOTION_CATEGORIES)),
                target_names=EMOTION_CATEGORIES,
                output_dict=True,
                zero_division=0
            )
            pd.DataFrame(report).transpose().to_csv(f"{RESULTS_DIR}/{name}_emotion_report.csv")
            joblib.dump(model, f"{MODELS_DIR}/{name}_emotion.joblib")

    # Save final results (only columns that exist)
    output_cols = ['index', 'comment', 'sentiment', 'emotion', 'clean_comment']
    for name in models.keys():
        output_cols.append(f'{name}_sentiment')
    output_cols.extend(emotion_cols)
    df[output_cols].to_csv(f"{RESULTS_DIR}/hmpv_ml_results.csv", index=False)
    print("\n✅ Analysis complete. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main()
