"""
vader_worker.py - the sentiment worker, kept in a tiny module so that each worker process
imports only re + nltk's VADER (not pandas/numpy), which keeps the memory of the pool small.
"""
import re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_VADER = None


def init():
    global _VADER
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()


def score_chunk(texts):
    """Mean VADER compound score over a review's sentences (at most 60 sentences). A whole-text
    compound saturates towards +/-1 on long reviews; the sentence mean separates reviews better."""
    out = []
    for t in texts:
        sents = [s for s in SENT_SPLIT.split(t) if s.strip()][:60]
        if sents:
            out.append(sum(_VADER.polarity_scores(s)["compound"] for s in sents) / len(sents))
        else:
            out.append(0.0)
    return out
