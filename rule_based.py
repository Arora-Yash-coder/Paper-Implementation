import pandas as pd
import numpy as np
import os
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from cleantext import clean

nltk.download('vader_lexicon')
nltk.download('stopwords')

from nltk.corpus import stopwords

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = clean(text,
                 fix_unicode=True,
                 to_ascii=True,
                 no_line_breaks=True,
                 no_urls=True,
                 no_emails=True,
                 no_phone_numbers=True,
                 no_numbers=True,
                 no_digits=True,
                 no_currency_symbols=True,
                 no_punct=True,
                 replace_with_url="",
                 replace_with_email="",
                 replace_with_phone_number="",
                 replace_with_number="",
                 replace_with_digit="",
                 replace_with_currency_symbol="")
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def vader_sentiment(text):
    if not text:
        return 0
    scores = SentimentIntensityAnalyzer().polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 1
    elif compound <= -0.05:
        return -1
    else:
        return 0

def textblob_sentiment(text):
    if not text:
        return 0
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return 1
    elif polarity < -0.05:
        return -1
    else:
        return 0

def evaluate_model(true_labels, predicted_labels, model_name, task, output_dir):
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    
    # Save report
    report_path = os.path.join(output_dir, f"{model_name}_{task}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{model_name} {task.capitalize()} Classification Report:\n\n")
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")

    return accuracy

def main():
    input_file = "input_data.csv"
    output_dir = "results/rule_based_models"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)
    df.dropna(subset=['comment', 'sentiment', 'emotion'], inplace=True)
    df['comment'] = df['comment'].apply(preprocess_text)

    df['sentiment_encoded'] = df['sentiment'].astype(int)

    emotion_encoder = LabelEncoder()
    df['emotion_encoded'] = emotion_encoder.fit_transform(df['emotion'])
    np.save(os.path.join(output_dir, 'emotion_classes.npy'), emotion_encoder.classes_)

    df['vader_sentiment'] = df['comment'].apply(vader_sentiment)
    df['textblob_sentiment'] = df['comment'].apply(textblob_sentiment)

    # Sentiment Evaluation
    vader_acc = evaluate_model(df['sentiment_encoded'], df['vader_sentiment'],
                               model_name='VADER', task='sentiment', output_dir=output_dir)
    textblob_acc = evaluate_model(df['sentiment_encoded'], df['textblob_sentiment'],
                                  model_name='TextBlob', task='sentiment', output_dir=output_dir)

    # Log accuracies
    log_file = os.path.join(output_dir, "accuracies.txt")
    with open(log_file, 'w') as f:
        f.write("Rule-Based Sentiment Model Accuracies\n")
        f.write(f"VADER Accuracy: {vader_acc:.4f}\n")
        f.write(f"TextBlob Accuracy: {textblob_acc:.4f}\n")

    # Save predictions
    df.to_csv(os.path.join(output_dir, "rule_based_predictions.csv"), index=False)
    print("Rule-based sentiment analysis complete. Results saved.")

if __name__ == "__main__":
    main()
