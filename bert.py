# bert_only.py
import os
os.environ["USE_TF"] = "0"

import re
import nltk
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

nltk.download("stopwords")
nltk.download("wordnet")

# ========== CONFIG ==========
INPUT_FILE = "input_data.csv"
OUTPUT_DIR = "results/bert_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== TEXT CLEANING ==========
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^\w\s]", "", str(text).lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ========== TRAINER HELPER ==========
def run_bert(task_name, texts, labels, label_encoder):
    print(f"\n--- BERT - {task_name} ---")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Hugging Face Dataset
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    encoded = dataset.map(tokenize, batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_encoder.classes_)
    )

    # Training arguments
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{task_name.lower()}"),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="epoch",      # ✅ fix for your version
        logging_strategy="epoch",
        save_strategy="no",
        disable_tqdm=False,
    )


    trainer = Trainer(model=model, args=args, train_dataset=encoded, eval_dataset=encoded)
    trainer.train()

    # ---- Training accuracy ----
    train_preds = trainer.predict(encoded).predictions.argmax(axis=1)
    train_acc = accuracy_score(labels, train_preds)

    y_true = label_encoder.inverse_transform(labels)
    y_pred = label_encoder.inverse_transform(train_preds)

    report = classification_report(y_true, y_pred, zero_division=0)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(report)

    # Save model + encoder
    model.save_pretrained(os.path.join(OUTPUT_DIR, f"{task_name.lower()}"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, f"{task_name.lower()}_encoder.joblib"))

    # Save predictions
    pred_df = pd.DataFrame({"text": texts, "actual": y_true, "predicted": y_pred})
    pred_file = os.path.join(OUTPUT_DIR, f"predictions_{task_name.lower()}.csv")
    pred_df.to_csv(pred_file, index=False)
    print(f"Saved predictions -> {pred_file}")

    return train_acc

# ========== MAIN ==========
def main():
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=["comment", "sentiment", "emotion"], inplace=True)
    df["clean_text"] = df["comment"].astype(str).apply(preprocess)

    # --- Sentiment ---
    le_sentiment = LabelEncoder()
    y_sentiment = le_sentiment.fit_transform(df["sentiment"])
    joblib.dump(le_sentiment, os.path.join(OUTPUT_DIR, "sentiment_encoder.joblib"))

    sentiment_acc = run_bert("Sentiment", df["clean_text"].tolist(), y_sentiment, le_sentiment)

    # --- Emotion ---
    le_emotion = LabelEncoder()
    y_emotion = le_emotion.fit_transform(df["emotion"])
    joblib.dump(le_emotion, os.path.join(OUTPUT_DIR, "emotion_encoder.joblib"))

    emotion_acc = run_bert("Emotion", df["clean_text"].tolist(), y_emotion, le_emotion)

    print("\n==== Final Results ====")
    print(f"Sentiment Training Accuracy: {sentiment_acc:.4f}")
    print(f"Emotion Training Accuracy: {emotion_acc:.4f}")

if __name__ == "__main__":
    main()
