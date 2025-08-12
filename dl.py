import os
import re
import nltk
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout

nltk.download('stopwords')
nltk.download('wordnet')

# ========== CONFIG ========== #
INPUT_FILE = 'input_data.csv'
OUTPUT_DIR = 'results/dl_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== TEXT CLEANING ========== #
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ========== MODEL BUILDERS ========== #
def build_cnn_model(input_length, vocab_size, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_lstm_model(input_length, vocab_size, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        LSTM(128),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_predict_save(model, model_name, task_name, X, y, label_encoder, raw_texts):
    history = model.fit(X, y, epochs=5, batch_size=64, validation_split=0.1, verbose=0)

    predictions = model.predict(X, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    acc = accuracy_score(y, y_pred)

    y_decoded = label_encoder.inverse_transform(y)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    report = classification_report(y_decoded, y_pred_decoded, zero_division=0)
    print(f"{model_name} - {task_name} - Accuracy: {acc:.4f}")
    print(report)

    # Save model
    model.save(os.path.join(OUTPUT_DIR, f"{model_name.lower()}_{task_name.lower()}.h5"))

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

    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    padded = pad_sequences(sequences, maxlen=100)

    vocab_size = min(len(tokenizer.word_index) + 1, 10000)
    input_length = padded.shape[1]

    joblib.dump(tokenizer, os.path.join(OUTPUT_DIR, "tokenizer.joblib"))

    # Sentiment
    le_sentiment = LabelEncoder()
    y_sentiment = le_sentiment.fit_transform(df['sentiment'])
    joblib.dump(le_sentiment, os.path.join(OUTPUT_DIR, "sentiment_encoder.joblib"))

    print("\n--- Sentiment Classification (Full Dataset) ---")
    for builder, name in [(build_cnn_model, "CNN"), (build_lstm_model, "LSTM")]:
        model = builder(input_length, vocab_size, len(le_sentiment.classes_))
        train_predict_save(model, name, "Sentiment", padded, y_sentiment, le_sentiment, df['comment'].values)

    # Emotion
    le_emotion = LabelEncoder()
    y_emotion = le_emotion.fit_transform(df['emotion'])
    joblib.dump(le_emotion, os.path.join(OUTPUT_DIR, "emotion_encoder.joblib"))

    print("\n--- Emotion Classification (Full Dataset) ---")
    for builder, name in [(build_cnn_model, "CNN"), (build_lstm_model, "LSTM")]:
        model = builder(input_length, vocab_size, len(le_emotion.classes_))
        train_predict_save(model, name, "Emotion", padded, y_emotion, le_emotion, df['comment'].values)

if __name__ == "__main__":
    main()
