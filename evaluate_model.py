import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentiment_model import predict_sentiment   # reuse existing logic

LABEL_MAP = {
    "Positive": 2,
    "Neutral": 1,
    "Negative": 0
}

REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}

def evaluate():
    df = pd.read_csv("data/test_data.csv")

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        text = row["text"]
        true_label = LABEL_MAP[row["label"]]

        pred_label = predict_sentiment(text, text)
        pred_label = LABEL_MAP[pred_label]

        y_true.append(true_label)
        y_pred.append(pred_label)

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation completed.")
    print(metrics)


if __name__ == "__main__":
    evaluate()
