# DEEP LEARNING SENTIMENT & EMOTION ANALYSIS (FIXED)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re
import os

# Configuration
DATA_PATH = "data/hmpv_comments_labeled.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"
EMOTION_CATEGORIES = [
    'happy', 'joy', 'fear', 'anger', 'enthusiasm',
    'sad', 'relief', 'sympathy', 'surprise', 'disgust', 'unemotional'
]

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', str(text).lower())
    return re.sub(r'[^a-zA-Z ]', '', text).strip()

def create_cnn_model(vocab_size, input_length, output_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128, input_length=input_length),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_units, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

def create_lstm_model(vocab_size, input_length, output_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128, input_length=input_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_units, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

def main():
    # Setup directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load and preprocess data
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Clean and validate data
    print("🧹 Cleaning and validating data...")
    df['clean_comment'] = df['comment'].apply(clean_text)
    
    # Handle sentiment labels
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(0).astype(int)
    df = df[df['sentiment'].isin([-1, 0, 1])]
    
    # Convert to 0-based labels
    df['sentiment_encoded'] = df['sentiment'] + 1  # Now 0,1,2
    print("✅ Sentiment distribution:", df['sentiment_encoded'].value_counts())

    # Handle emotion labels
    df['emotion'] = df['emotion'].where(df['emotion'].isin(EMOTION_CATEGORIES), 'unemotional')
    le_emotion = LabelEncoder()
    le_emotion.fit(EMOTION_CATEGORIES)
    df['emotion_encoded'] = le_emotion.transform(df['emotion'])

    # Tokenization
    print("🔠 Tokenizing comments...")
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_comment'])
    sequences = tokenizer.texts_to_sequences(df['clean_comment'])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    # Save preprocessing artifacts
    joblib.dump(tokenizer, f"{MODELS_DIR}/dl_tokenizer.joblib")
    joblib.dump(le_emotion, f"{MODELS_DIR}/dl_emotion_encoder.joblib")

    # ===== Sentiment Analysis =====
    print("\n🔍 Training Sentiment Models...")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        padded_sequences, 
        df['sentiment_encoded'],
        test_size=0.2,
        stratify=df['sentiment_encoded'],
        random_state=42
    )
    
    # CNN for Sentiment
    print("\n🚀 Training CNN for Sentiment...")
    cnn_sentiment = create_cnn_model(5000, 100, 3)
    cnn_sentiment.fit(X_train_s, y_train_s, 
                     epochs=5, 
                     batch_size=64,
                     validation_data=(X_test_s, y_test_s))
    cnn_sentiment.save(f"{MODELS_DIR}/cnn_sentiment.h5")
    
    # LSTM for Sentiment
    print("\n🚀 Training LSTM for Sentiment...")
    lstm_sentiment = create_lstm_model(5000, 100, 3)
    lstm_sentiment.fit(X_train_s, y_train_s, 
                      epochs=5, 
                      batch_size=64,
                      validation_data=(X_test_s, y_test_s))
    lstm_sentiment.save(f"{MODELS_DIR}/lstm_sentiment.h5")

    # ===== Emotion Analysis =====
    print("\n🎭 Training Emotion Models...")
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        padded_sequences, 
        df['emotion_encoded'],
        test_size=0.2,
        stratify=df['emotion_encoded'],
        random_state=42
    )
    
    # CNN for Emotion
    print("\n🚀 Training CNN for Emotion...")
    cnn_emotion = create_cnn_model(5000, 100, len(EMOTION_CATEGORIES))
    cnn_emotion.fit(X_train_e, y_train_e, 
                   epochs=5, 
                   batch_size=64,
                   validation_data=(X_test_e, y_test_e))
    cnn_emotion.save(f"{MODELS_DIR}/cnn_emotion.h5")
    
    # LSTM for Emotion
    print("\n🚀 Training LSTM for Emotion...")
    lstm_emotion = create_lstm_model(5000, 100, len(EMOTION_CATEGORIES))
    lstm_emotion.fit(X_train_e, y_train_e, 
                    epochs=5, 
                    batch_size=64,
                    validation_data=(X_test_e, y_test_e))
    lstm_emotion.save(f"{MODELS_DIR}/lstm_emotion.h5")

    # Generate and save sentiment classification reports
    print("\n📊 Generating sentiment model reports...")
    # CNN Sentiment Report
    y_pred_cnn_sent = np.argmax(cnn_sentiment.predict(X_test_s), axis=1)
    cnn_sent_report = classification_report(
        y_test_s - 1,  # Convert back to -1,0,1 format
        y_pred_cnn_sent - 1,  # Convert predictions back to -1,0,1 format
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )
    pd.DataFrame(cnn_sent_report).transpose().to_csv(f"{RESULTS_DIR}/cnn_sentiment_report.csv")

    # LSTM Sentiment Report
    y_pred_lstm_sent = np.argmax(lstm_sentiment.predict(X_test_s), axis=1)
    lstm_sent_report = classification_report(
        y_test_s - 1,  # Convert back to -1,0,1 format
        y_pred_lstm_sent - 1,  # Convert predictions back to -1,0,1 format
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )
    pd.DataFrame(lstm_sent_report).transpose().to_csv(f"{RESULTS_DIR}/lstm_sentiment_report.csv")

    # Generate and save emotion classification reports
    print("\n📊 Generating emotion model reports...")
    # CNN Emotion Report
    y_pred_cnn_emo = np.argmax(cnn_emotion.predict(X_test_e), axis=1)
    cnn_emo_report = classification_report(
        y_test_e,
        y_pred_cnn_emo,
        target_names=EMOTION_CATEGORIES,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(cnn_emo_report).transpose().to_csv(f"{RESULTS_DIR}/cnn_emotion_report.csv")

    # LSTM Emotion Report
    y_pred_lstm_emo = np.argmax(lstm_emotion.predict(X_test_e), axis=1)
    lstm_emo_report = classification_report(
        y_test_e,
        y_pred_lstm_emo,
        target_names=EMOTION_CATEGORIES,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(lstm_emo_report).transpose().to_csv(f"{RESULTS_DIR}/lstm_emotion_report.csv")

    # Generate predictions
    print("\n🔍 Generating predictions...")
    df['cnn_sentiment'] = np.argmax(cnn_sentiment.predict(padded_sequences), axis=1) - 1
    df['lstm_sentiment'] = np.argmax(lstm_sentiment.predict(padded_sequences), axis=1) - 1
    df['cnn_emotion'] = le_emotion.inverse_transform(np.argmax(cnn_emotion.predict(padded_sequences), axis=1))
    df['lstm_emotion'] = le_emotion.inverse_transform(np.argmax(lstm_emotion.predict(padded_sequences), axis=1))

    # Save results
    output_cols = [
        'index', 'comment', 'sentiment', 'emotion', 'clean_comment',
        'cnn_sentiment', 'lstm_sentiment', 'cnn_emotion', 'lstm_emotion'
    ]
    df[output_cols].to_csv(f"{RESULTS_DIR}/hmpv_dl_results.csv", index=False)
    print("\n✅ Analysis complete. Results saved to 'results/' directory")

if __name__ == "__main__":
    main()
