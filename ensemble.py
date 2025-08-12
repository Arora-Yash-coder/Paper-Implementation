# ensemble.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# -----------------------
# Paths
# -----------------------
ROOT = "."
ML_DIR = os.path.join(ROOT, "results", "ml_models")
DL_DIR = os.path.join(ROOT, "results", "dl_models")
RULE_DIR = os.path.join(ROOT, "results", "rule_based_models")
ENS_DIR = os.path.join(ROOT, "results", "ensemble_models")

os.makedirs(ENS_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def ensure_text_column(df):
    """Rename 'comment' to 'text' if needed and return df."""
    if 'text' not in df.columns and 'comment' in df.columns:
        df = df.rename(columns={'comment': 'text'})
    return df

def check_alignment(base_series, other_series, name):
    """Raise error if text series differ."""
    if not base_series.equals(other_series):
        raise ValueError(f"Text alignment mismatch with {name} file. Fix CSV ordering or use a merge by key.")

def train_and_save_lightgbm(X_df, y_series, out_texts, outfile_csv, outfile_model):
    """
    Train LightGBM on full data, save predictions CSV and pickle model+labelencoder.
    Returns accuracy (float).
    """
    # Fit label encoder on union of y + feature values (stringified)
    all_values = pd.concat([y_series.astype(str)] + [X_df[c].astype(str) for c in X_df.columns], ignore_index=True)
    le = LabelEncoder().fit(all_values)

    y_enc = le.transform(y_series.astype(str))
    X_enc = X_df.copy()
    for c in X_enc.columns:
        X_enc[c] = le.transform(X_enc[c].astype(str))

    model = LGBMClassifier(random_state=42)
    model.fit(X_enc, y_enc)

    preds_enc = model.predict(X_enc)
    preds = le.inverse_transform(preds_enc)
    acc = accuracy_score(y_enc, preds_enc)

    out_df = pd.DataFrame({
        "text": out_texts,
        "actual": y_series.astype(str),
        "ensemble_predicted": preds
    })
    out_df.to_csv(outfile_csv, index=False)

    # Save model + label encoder + feature columns
    joblib.dump({
        "model": model,
        "label_encoder": le,
        "feature_columns": list(X_df.columns)
    }, outfile_model)

    return acc

# -----------------------
# Sentiment ensemble
# -----------------------
def build_sentiment_ensemble():
    print("== Building Sentiment Ensemble ==")

    # mapping: model_name -> (full_path_to_csv, column_name_in_csv, accuracy_estimate)
    sentiment_models = {
        "SVM": (os.path.join(ML_DIR, "predictions_svm_sentiment.csv"), "predicted", 0.7037),
        "LSTM": (os.path.join(DL_DIR, "predictions_lstm_sentiment.csv"), "predicted", 0.8648),
        "CNN": (os.path.join(DL_DIR, "predictions_cnn_sentiment.csv"), "predicted", 0.9148),
        "NaiveBayes": (os.path.join(ML_DIR, "predictions_naivebayes_sentiment.csv"), "predicted", 0.6238),
        "RandomForest": (os.path.join(ML_DIR, "predictions_randomforest_sentiment.csv"), "predicted", 0.9290),
        "Vader": (os.path.join(RULE_DIR, "rule_based_predictions.csv"), "vader_sentiment", 0.4661),
        "TextBlob": (os.path.join(RULE_DIR, "rule_based_predictions.csv"), "textblob_sentiment", 0.4152),
    }

    # priority for tie-breaking (higher priority earlier)
    model_priority = ["RandomForest", "CNN", "LSTM", "SVM", "NaiveBayes", "Vader", "TextBlob"]

    # read CSVs, collect predictions
    preds_dict = {}
    texts = None
    y_true = None

    for mname, (path, col, _) in sentiment_models.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sentiment CSV not found: {path}")
        df = pd.read_csv(path)
        df = ensure_text_column(df)

        if texts is None:
            if 'text' in df.columns:
                texts = df['text']
        else:
            # verify alignment by text column
            if 'text' in df.columns:
                check_alignment(texts, df['text'], mname)

        # find actual labels (take from first available CSV)
        if y_true is None:
            for candidate in ('actual', 'true_label', 'sentiment'):
                if candidate in df.columns:
                    y_true = df[candidate]
                    break

        # extract predictions column
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in {path}")
        preds_dict[mname] = df[col].astype(str)

    if texts is None:
        raise ValueError("Cannot determine 'text' column from sentiment CSVs.")
    if y_true is None:
        raise ValueError("No actual/true_label/sentiment column found in any sentiment CSV.")

    # create DataFrame of base-model predictions
    preds_df = pd.DataFrame(preds_dict)

    # accuracy weights dict
    accuracies = {m: acc for m, (_, _, acc) in sentiment_models.items()}

    # Weighted voting (accuracy-weighted) with tie-breaking by priority
    weighted_votes = []
    for _, row in preds_df.iterrows():
        vote_scores = {}
        for model_name in preds_df.columns:
            pred_label = row[model_name]
            vote_scores[pred_label] = vote_scores.get(pred_label, 0) + accuracies.get(model_name, 1)
        max_score = max(vote_scores.values())
        winners = [lbl for lbl, score in vote_scores.items() if score == max_score]
        if len(winners) == 1:
            weighted_votes.append(winners[0])
        else:
            chosen = None
            for p in model_priority:
                if row[p] in winners:
                    chosen = row[p]
                    break
            weighted_votes.append(chosen if chosen is not None else winners[0])

    # Save weighted voting CSV
    weighted_csv = os.path.join(ENS_DIR, "ensemble_weighted_sentiment.csv")
    pd.DataFrame({"text": texts, "actual": y_true.astype(str), "ensemble_predicted": weighted_votes}).to_csv(weighted_csv, index=False)
    print(f"Saved weighted-vote ensemble CSV -> {weighted_csv}")

    # Train LightGBM ensemble on full data and save model+csv
    lgb_csv = os.path.join(ENS_DIR, "ensemble_lightgbm_sentiment.csv")
    lgb_model_path = os.path.join(ENS_DIR, "ensemble_lightgbm_sentiment.pkl")
    acc = train_and_save_lightgbm(preds_df, y_true, texts, lgb_csv, lgb_model_path)
    print(f"Trained LightGBM Sentiment ensemble. Accuracy (on full data): {acc*100:.2f}%")
    print(f"Saved LightGBM CSV -> {lgb_csv}")
    print(f"Saved LightGBM model -> {lgb_model_path}")

# -----------------------
# Emotion ensemble
# -----------------------
def build_emotion_ensemble():
    print("== Building Emotion Ensemble ==")

    emotion_models = {
        "SVM": (os.path.join(ML_DIR, "predictions_svm_emotion.csv"), "predicted", 0.5558),
        "LSTM": (os.path.join(DL_DIR, "predictions_lstm_emotion.csv"), "predicted", 0.6936),
        "CNN": (os.path.join(DL_DIR, "predictions_cnn_emotion.csv"), "predicted", 0.7574),
        "NaiveBayes": (os.path.join(ML_DIR, "predictions_naivebayes_emotion.csv"), "predicted", 0.3973),
        "RandomForest": (os.path.join(ML_DIR, "predictions_randomforest_emotion.csv"), "predicted", 0.9042)
    }

    model_priority = ["RandomForest", "CNN", "LSTM", "SVM", "NaiveBayes"]

    preds_dict = {}
    texts = None
    y_true = None

    for mname, (path, col, _) in emotion_models.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Emotion CSV not found: {path}")
        df = pd.read_csv(path)
        df = ensure_text_column(df)

        if texts is None:
            if 'text' in df.columns:
                texts = df['text']
        else:
            if 'text' in df.columns:
                check_alignment(texts, df['text'], mname)

        if y_true is None:
            for candidate in ('actual', 'true_label', 'emotion'):
                if candidate in df.columns:
                    y_true = df[candidate]
                    break

        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in {path}")
        preds_dict[mname] = df[col].astype(str)

    if texts is None:
        raise ValueError("Cannot determine 'text' column from emotion CSVs.")
    if y_true is None:
        raise ValueError("No actual/true_label/emotion column found in any emotion CSV.")

    preds_df = pd.DataFrame(preds_dict)

    accuracies = {m: acc for m, (_, _, acc) in emotion_models.items()}

    weighted_votes = []
    for _, row in preds_df.iterrows():
        vote_scores = {}
        for model_name in preds_df.columns:
            pred_label = row[model_name]
            vote_scores[pred_label] = vote_scores.get(pred_label, 0) + accuracies.get(model_name, 1)
        max_score = max(vote_scores.values())
        winners = [lbl for lbl, score in vote_scores.items() if score == max_score]
        if len(winners) == 1:
            weighted_votes.append(winners[0])
        else:
            chosen = None
            for p in model_priority:
                if row[p] in winners:
                    chosen = row[p]
                    break
            weighted_votes.append(chosen if chosen is not None else winners[0])

    weighted_csv = os.path.join(ENS_DIR, "ensemble_weighted_emotion.csv")
    pd.DataFrame({"text": texts, "actual": y_true.astype(str), "ensemble_predicted": weighted_votes}).to_csv(weighted_csv, index=False)
    print(f"Saved weighted-vote ensemble CSV -> {weighted_csv}")

    lgb_csv = os.path.join(ENS_DIR, "ensemble_lightgbm_emotion.csv")
    lgb_model_path = os.path.join(ENS_DIR, "ensemble_lightgbm_emotion.pkl")
    acc = train_and_save_lightgbm(preds_df, y_true, texts, lgb_csv, lgb_model_path)
    print(f"Trained LightGBM Emotion ensemble. Accuracy (on full data): {acc*100:.2f}%")
    print(f"Saved LightGBM CSV -> {lgb_csv}")
    print(f"Saved LightGBM model -> {lgb_model_path}")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    build_sentiment_ensemble()
    build_emotion_ensemble()
