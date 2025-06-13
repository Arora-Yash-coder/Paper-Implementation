import os
import re
import nltk
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
def build_cnn_model(input_length, vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # for binary, will be replaced
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

def build_lstm_model(input_length, vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # for binary, will be replaced
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

# ========== TRAIN & SAVE FUNCTION ========== #
def train_evaluate_save(model, model_name, task_name, X_train, X_test, y_train, y_test, label_encoder, raw_texts_test):
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=0)

    train_acc = history.history['accuracy'][-1]
    test_preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(test_preds, axis=1)

    test_acc = accuracy_score(y_test, y_pred)

    # Decode for human-readable output
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    class_names = [str(c) for c in label_encoder.classes_]

    report = classification_report(
        y_test_decoded, y_pred_decoded,
        labels=class_names,
        target_names=class_names,
        zero_division=0
    )

    print(f"{model_name} - {task_name.lower()} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(report)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_{task_name.lower()}.h5")
    model.save(model_path)

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

    # Tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    padded = pad_sequences(sequences, maxlen=100)

    # Save tokenizer ✅
    joblib.dump(tokenizer, os.path.join(OUTPUT_DIR, "tokenizer.joblib"))

    # Label encoders
    le_sentiment = LabelEncoder()
    y_sentiment = le_sentiment.fit_transform(df['sentiment'])
    joblib.dump(le_sentiment, os.path.join(OUTPUT_DIR, "sentiment_encoder.joblib"))  # ✅

    le_emotion = LabelEncoder()
    y_emotion = le_emotion.fit_transform(df['emotion'])
    joblib.dump(le_emotion, os.path.join(OUTPUT_DIR, "emotion_encoder.joblib"))  # ✅

    # --- Sentiment Classification --- #
    print("\n--- Sentiment Classification ---")
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        padded, y_sentiment, df['comment'].values, test_size=0.2, random_state=42
    )

    vocab_size = min(len(tokenizer.word_index) + 1, 10000)
    input_length = padded.shape[1]

    for builder, name in [(build_cnn_model, "CNN"), (build_lstm_model, "LSTM")]:
        model = builder(input_length, vocab_size)
        model.pop()
        model.add(Dense(len(le_sentiment.classes_), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_evaluate_save(model, name, "Sentiment", X_train, X_test,
                            y_train, y_test, le_sentiment, texts_test)

    # --- Emotion Classification --- #
    print("\n--- Emotion Classification ---")
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        padded, y_emotion, df['comment'].values, test_size=0.2, random_state=42
    )

    for builder, name in [(build_cnn_model, "CNN"), (build_lstm_model, "LSTM")]:
        model = builder(input_length, vocab_size)
        model.pop()
        model.add(Dense(len(le_emotion.classes_), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_evaluate_save(model, name, "Emotion", X_train, X_test,
                            y_train, y_test, le_emotion, texts_test)

if __name__ == "__main__":
    main()
