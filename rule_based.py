# RULE-BASED SENTIMENT ANALYSIS (Updated)
import pandas as pd
import re
import ast
import os
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv("hmpv_tweets.csv")

print("🔄 Exploding comments...")
df['comments'] = df['comments'].apply(lambda x: ast.literal_eval(x))

comments_data = []
for idx, row in df.iterrows():
    for comment in row['comments']:
        comments_data.append({
            'post_index': idx,
            'comment': comment
        })

comments_df = pd.DataFrame(comments_data)

print("🧹 Cleaning comments...")
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', text.lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

comments_df['clean_comment'] = comments_df['comment'].apply(clean_text)

print("🧠 Initializing sentiment analyzers...")
analyzer = SentimentIntensityAnalyzer()

print("🔍 Applying VADER sentiment analysis...")
comments_df['vader_score'] = comments_df['clean_comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
comments_df['vader_sentiment'] = comments_df['vader_score'].apply(lambda x: -1 if x < -0.05 else (1 if x > 0.05 else 0))

print("🔍 Applying TextBlob sentiment analysis...")
comments_df['textblob_score'] = comments_df['clean_comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
comments_df['textblob_sentiment'] = comments_df['textblob_score'].apply(lambda x: -1 if x < -0.1 else (1 if x > 0.1 else 0))

# Save results
output_file = "hmpv_rule_based_results.csv"
comments_df.to_csv(output_file, index=False)
print(f"✅ Rule-based results saved to {output_file}")

# Save models
joblib.dump(analyzer, "models/vader_analyzer.joblib")
joblib.dump(TextBlob, "models/textblob_sentiment_function.joblib")  # Function-like storage
print("💾 Sentiment analyzers saved to 'models/' folder")
