# RULE-BASED SENTIMENT ANALYSIS
import pandas as pd
import re
import os
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import classification_report

# Configure paths
DATA_PATH = "data/hmpv_comments_labeled.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def clean_text(text):
    """Clean and normalize text data"""
    text = re.sub(r"http\S+|www\S+", '', str(text).lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

def validate_sentiment(df):
    """Clean and validate sentiment labels"""
    # Convert to numeric and handle invalid values
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    # Fill NaN with neutral (0) and convert to integers
    df['sentiment'] = df['sentiment'].fillna(0).astype(int)
    # Ensure values are within [-1, 0, 1]
    df['sentiment'] = df['sentiment'].apply(lambda x: x if x in {-1, 0, 1} else 0)
    return df

def main():
    # Load and preprocess data
    print("📥 Loading labeled dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Clean and validate sentiment labels
    df = validate_sentiment(df)
    
    # Clean comments
    print("🧹 Cleaning comments...")
    df['clean_comment'] = df['comment'].apply(clean_text)

    # Initialize analyzers
    print("🧠 Initializing sentiment analyzers...")
    vader_analyzer = SentimentIntensityAnalyzer()

    # VADER Analysis
    print("🔍 Applying VADER analysis...")
    df['vader_score'] = df['clean_comment'].apply(
        lambda x: vader_analyzer.polarity_scores(x)['compound']
    )
    df['vader_sentiment'] = df['vader_score'].apply(
        lambda x: -1 if x < -0.05 else (1 if x > 0.05 else 0)
    ).astype(int)

    # TextBlob Analysis
    print("🔍 Applying TextBlob analysis...")
    df['textblob_score'] = df['clean_comment'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df['textblob_sentiment'] = df['textblob_score'].apply(
        lambda x: -1 if x < -0.1 else (1 if x > 0.1 else 0)
    ).astype(int)

    # Generate reports
    print("📊 Generating classification reports...")
    
    # VADER report
    vader_report = classification_report(
        df['sentiment'], 
        df['vader_sentiment'],
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )
    pd.DataFrame(vader_report).transpose().to_csv(
        f"{RESULTS_DIR}/vader_classification_report.csv"
    )

    # TextBlob report
    textblob_report = classification_report(
        df['sentiment'], 
        df['textblob_sentiment'],
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )
    pd.DataFrame(textblob_report).transpose().to_csv(
        f"{RESULTS_DIR}/textblob_classification_report.csv"
    )

    # Save results
    output_cols = [
        'index', 'comment', 'sentiment', 'clean_comment',
        'vader_score', 'vader_sentiment',
        'textblob_score', 'textblob_sentiment'
    ]
    
    df[output_cols].to_csv(
        f"{RESULTS_DIR}/hmpv_sentiment_results.csv", index=False
    )
    
    # Save models
    joblib.dump(vader_analyzer, f"{MODELS_DIR}/vader_analyzer.joblib")
    joblib.dump(TextBlob, f"{MODELS_DIR}/textblob_analyzer.joblib")
    
    print("✅ Analysis complete. Results saved to 'results/' directory")

if __name__ == "__main__":
    main()
