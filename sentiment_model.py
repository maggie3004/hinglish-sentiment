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
        probs = F.softmax(outputs.logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    sentiment = LABELS[pred_idx]
    confidence = probs[pred_idx].item()

    # -------- Emoji logic --------
    emoji_score = emoji_sentiment_score(original_text)
    if emoji_score >= 1 and sentiment != "Negative":
        sentiment = "Positive"
    elif emoji_score <= -1:
        sentiment = "Negative"

    # -------- Sarcasm --------
    if detect_sarcasm(original_text) and sentiment == "Positive":
        sentiment = "Negative"

    # -------- Negation merged with Negative --------
    if detect_negation(original_text):
        sentiment = "Negative"

    # -------- Short neutral text --------
    if sentiment == "Positive" and emoji_score == 0 and len(clean_text.split()) <= 2:
        sentiment = "Neutral"

    return sentiment, confidence, probs.tolist()
