import re

POSITIVE_EMOJIS = ["😀", "😃", "😄", "😁", "😊", "😍", "👍", "❤️", "🔥"]
NEGATIVE_EMOJIS = ["😡", "😠", "😞", "😢", "👎", "💔", "😭"]

NEGATION_WORDS = [
    "nhi", "nahin", "nahi", "not", "never", "no", "dont", "didnt", "doesnt"
]


def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def emoji_sentiment_score(text):
    score = 0
    for e in POSITIVE_EMOJIS:
        if e in text:
            score += 1
    for e in NEGATIVE_EMOJIS:
        if e in text:
            score -= 1
    return score


def detect_sarcasm(text):
    sarcasm_words = ["wah", "great", "nice", "amazing"]
    text = text.lower()
    return "!" in text and any(w in text for w in sarcasm_words)


def detect_negation(text):
    words = text.lower().split()
    return any(word in words for word in NEGATION_WORDS)
