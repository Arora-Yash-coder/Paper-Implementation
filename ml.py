# ML-BASED SENTIMENT ANALYSIS (Updated)
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

print("📥 Loading rule-based sentiment results...")
df = pd.read_csv("hmpv_rule_based_results.csv")

print("⚙️ Generating confidence-weighted sentiment...")
df['combined_score'] = (df['vader_score'] + df['textblob_score']) / 2

def assign_sentiment(score):
    if score > 0.05:
        return 1   # Positive
    elif score < -0.05:
        return -1  # Negative
    else:
        return 0   # Neutral

df['sentiment'] = df['combined_score'].apply(assign_sentiment)

print(f"🧾 Total samples: {len(df)}")
print(df['sentiment'].value_counts())

# Vectorize
print("🔠 Vectorizing clean comments...")
df['clean_comment'] = df['clean_comment'].fillna('')
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df['clean_comment'])
y = df['sentiment']

# Save vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("✅ TF-IDF vectorizer saved to 'models/tfidf_vectorizer.joblib'")

# Split
print("🧪 Splitting train/test data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "naive_bayes": MultinomialNB(),
    "svm": SVC(kernel='linear', probability=True, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
}

results_df = df.copy()

# Train and save models
for name, model in models.items():
    print(f"\n🚀 Training {name.replace('_', ' ').title()}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"📊 {name.replace('_', ' ').title()} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4))

    # Save model
    joblib.dump(model, f"models/{name}_model.joblib")
    print(f"💾 Saved to models/{name}_model.joblib")

    # Predict on full data
    results_df[f'{name}_prediction'] = model.predict(X)

# Save results
output_file = "hmpv_ml_results.csv"
results_df.to_csv(output_file, index=False)
print(f"\n📁 ML predictions saved to '{output_file}'")
