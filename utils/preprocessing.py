import re

POSITIVE_EMOJIS = [
    "😀", "😃", "😄", "😁", "😊", "☺️",
    "😍", "🥰", "😘", "😇",
    "👍", "👌", "👏", "🙌",
    "❤️", "💖", "💕", "💞", "💓",
    "🔥", "✨", "🎉", "😎"
]


NEGATIVE_EMOJIS = [
    "😡", "😠", "🤬",
    "😞", "😔", "😢", "😭", "😩", "😫",
    "👎", "💔",
    "😒", "🙄", "😤",
    "☹️", "😕"
]


NEGATION_WORDS = [
    # Hindi / Hinglish
    "nhi", "nai", "nahi", "nahin",
    "na", "mat", "bilkul nahi",

    # English
    "not", "no", "never", "none", "nothing",

    # Contractions / informal
    "dont", "don't",
    "didnt", "didn't",
    "doesnt", "doesn't",
    "cant", "can't",
    "wont", "won't",
    "isnt", "isn't",
    "arent", "aren't",

    # Common Hinglish phrases
    "acha nahi",
    "theek nahi",
    "pasand nahi"
]


def clean_tweet(text):
    """
    Light cleaning only.
    Do NOT remove punctuation aggressively.
    """
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
    text = text.lower()
    sarcasm_words = ["wah", "great", "nice", "amazing"]
    return "!" in text and any(w in text for w in sarcasm_words)

def detect_negation(text):
    text = text.lower()
    for neg in NEGATION_WORDS:
        if neg in text:
            return True
    return False

