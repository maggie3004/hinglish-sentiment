import streamlit as st
import os
import json
import pandas as pd

from utils.preprocessing import (
    clean_tweet,
    emoji_sentiment_score,
    detect_sarcasm,
    detect_negation
)
from sentiment_model import predict_sentiment

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hinglish Sentiment Analysis",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("Hinglish Sentiment Analysis")

# ---------------- SIDEBAR ----------------
st.sidebar.title("About Project")
st.sidebar.write("""
• Hinglish BERT based sentiment analysis  
• Emoji sentiment enhancement  
• Sarcasm handling  
• Negation handling
""")

st.sidebar.markdown("---")
st.sidebar.caption(
    "This system combines a pretrained Hinglish-adapted transformer "
    "with linguistic heuristics for better interpretability."
)

# ---------------- PERFORMANCE METRICS ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")

METRICS_FILE = "metrics.json"

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [
            metrics.get("accuracy"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1_score")
        ]
    })

    st.sidebar.table(metrics_df)
else:
    st.sidebar.warning("Performance metrics not available")

# ---------------- HELPER UI ----------------
def sentiment_card(sentiment):
    if sentiment == "Positive":
        st.success("😊 Positive Sentiment")
    elif sentiment == "Negative":
        st.error("😠 Negative Sentiment")
    else:
        st.info("😐 Neutral Sentiment")

# ---------------- EXAMPLES ----------------
st.subheader("Try Examples")

if "example" not in st.session_state:
    st.session_state.example = ""

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Positive"):
        st.session_state.example = "yeh movie bahut achi hai 😍"

with c2:
    if st.button("Negative"):
        st.session_state.example = "phone bilkul bekaar hai 😡"

with c3:
    if st.button("Neutral"):
        st.session_state.example = "movie theek tha"

# ---------------- INPUT ----------------
tweet = st.text_input(
    "Enter Hinglish Tweet",
    value=st.session_state.example,
    placeholder="Type Hinglish tweet here..."
)

st.caption(f"Characters: {len(tweet)} / 280")

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- ANALYSIS ----------------
if st.button("Analyze"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet")
    else:
        cleaned = clean_tweet(tweet)
        sentiment, confidence_model, probs = predict_sentiment(cleaned, tweet)

        # Save history
        st.session_state.history.append((tweet, sentiment))

        # Sentiment result
        sentiment_card(sentiment)

        # -------- SENTIMENT SCORES (ADDED) --------
        scores_df = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Score": [
                round(probs[0], 4),
                round(probs[1], 4),
                round(probs[2], 4)
            ]
        })

        st.subheader("Sentiment Scores")
        st.table(scores_df)

        # Emoji signal
        emoji_score = emoji_sentiment_score(tweet)
        st.progress(min(abs(emoji_score), 5) / 5)

        if emoji_score > 0:
            st.caption("Emoji tone looks positive")
        elif emoji_score < 0:
            st.caption("Emoji tone looks negative")
        else:
            st.caption("No strong emoji signal")

        # Linguistic signals
        if detect_negation(tweet):
            st.info("ℹ️ Negation detected in sentence")
        if detect_sarcasm(tweet):
            st.warning("⚠️ Sarcasm detected")

        # -------- WHY THIS RESULT --------
        with st.expander("Why this result?"):
            reasons = []
            if detect_negation(tweet):
                reasons.append("Negation words detected (nahi / nhi / not)")
            if detect_sarcasm(tweet):
                reasons.append("Sarcasm pattern detected")
            if emoji_score > 0:
                reasons.append("Positive emojis present")
            elif emoji_score < 0:
                reasons.append("Negative emojis present")

            if not reasons:
                reasons.append("Model prediction based on textual context")

            for r in reasons:
                st.write("•", r)

        # -------- CONFIDENCE (APPROX) --------
        confidence = 0.6
        if emoji_score != 0:
            confidence += 0.2
        if detect_negation(tweet) or detect_sarcasm(tweet):
            confidence += 0.1

        st.caption(f"Approximate model confidence: {int(confidence * 100)}%")

        # -------- DISCLAIMER --------
        st.markdown("""
        <small>
        ⚠️ Note: Hinglish sentiment analysis is challenging due to code-mixing,
        sarcasm, negation, and informal grammar. Results may vary in ambiguous cases.
        </small>
        """, unsafe_allow_html=True)

# ---------------- HISTORY VIEW ----------------
if st.session_state.history:
    with st.expander("Previous Predictions"):
        for t, s in st.session_state.history[-5:]:
            st.write(f"• {t} → {s}")

# ---------------- CLEAR ----------------
if st.button("Clear"):
    st.session_state.example = ""
    st.rerun()
