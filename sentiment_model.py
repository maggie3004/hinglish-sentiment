import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from utils.preprocessing import (
    emoji_sentiment_score,
    detect_sarcasm,
    detect_negation
)

MODEL_DIR = "models/hinglish_bert"

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-uncased"
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-uncased",
    num_labels=3
)

state_dict = torch.load(
    f"{MODEL_DIR}/pytorch_model_head.bin",
    map_location="cpu"
)
model.load_state_dict(state_dict, strict=False)
model.eval()

LABELS = ["Negative", "Neutral", "Positive"]

def predict_sentiment(clean_text, original_text):
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0].tolist()

    # Base model scores
    score_map = {
        "Negative": probs[0],
        "Neutral": probs[1],
        "Positive": probs[2]
    }

    # Linguistic signals
    emoji_score = emoji_sentiment_score(original_text)
    has_negation = detect_negation(original_text)
    has_sarcasm = detect_sarcasm(original_text)

    # ---------- Heuristic adjustments ----------
    if emoji_score <= -1:
        score_map["Negative"] += 0.25
        score_map["Positive"] -= 0.15

    if emoji_score >= 1:
        score_map["Positive"] += 0.25
        score_map["Negative"] -= 0.15

    if has_negation:
        score_map["Negative"] += 0.20
        score_map["Positive"] -= 0.10

    if has_sarcasm:
        score_map["Negative"] += 0.20
        score_map["Positive"] -= 0.10

    # ---------- Normalize ----------
    for k in score_map:
        score_map[k] = max(score_map[k], 0.0)

    total = sum(score_map.values())
    for k in score_map:
        score_map[k] /= total

    # Final decision
    sentiment = max(score_map, key=score_map.get)
    confidence = score_map[sentiment]

    probs_final = [
        score_map["Negative"],
        score_map["Neutral"],
        score_map["Positive"]
    ]

    return sentiment, confidence, probs_final
