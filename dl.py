import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import joblib

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load rule-based sentiment results
print("Loading rule-based sentiment results...")
df = pd.read_csv("hmpv_rule_based_results.csv")

# Generate Confidence Weighted Sentiment
print("Generating final sentiment labels (confidence weighted)...")
df['combined_score'] = (df['vader_score'] + df['textblob_score']) / 2

def assign_sentiment(score):
    if score > 0.05:
        return 2   # Positive
    elif score < -0.05:
        return 0   # Negative
    else:
        return 1   # Neutral

df['sentiment'] = df['combined_score'].apply(assign_sentiment)

print(f"Total samples: {len(df)}")
print(df['sentiment'].value_counts())

# Prepare text data
print("Preparing tokenized padded sequences...")
df['clean_comment'] = df['clean_comment'].fillna('')
texts = df['clean_comment'].values

# Tokenization
vocab_size = 5000
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Save tokenizer
tokenizer_path = "models/tokenizer_dl.pkl"
joblib.dump(tokenizer, tokenizer_path)
print(f"Tokenizer saved to {tokenizer_path}")

# Features and labels
X = padded_sequences
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
def create_cnn(vocab_size, input_length):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=input_length),
        Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define LSTM model
def create_lstm(vocab_size, input_length):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=input_length),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train CNN model
print("\nTraining CNN model...")
cnn_model = create_cnn(vocab_size, max_length)
cnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save CNN model
cnn_path = "models/cnn_model.h5"
cnn_model.save(cnn_path)
print(f"CNN model saved to {cnn_path}")

# Predict with CNN model
print("Predicting using CNN model...")
cnn_preds = cnn_model.predict(X)
df['cnn_sentiment'] = np.argmax(cnn_preds, axis=1)

# Train LSTM model
print("\nTraining LSTM model...")
lstm_model = create_lstm(vocab_size, max_length)
lstm_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save LSTM model
lstm_path = "models/lstm_model.h5"
lstm_model.save(lstm_path)
print(f"LSTM model saved to {lstm_path}")

# Predict with LSTM model
print("Predicting using LSTM model...")
lstm_preds = lstm_model.predict(X)
df['lstm_sentiment'] = np.argmax(lstm_preds, axis=1)

# Save results
output_file = "hmpv_dl_results.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Deep Learning sentiment analysis completed. Results saved to {output_file}")
