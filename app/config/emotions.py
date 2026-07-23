"""Canonical Ekman-6 + neutral emotion space (single source of truth)."""
from enum import Enum


class Emotion(str, Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


# Fixed order used in every table, confusion matrix and figure.
CANONICAL_ORDER: list[Emotion] = [
    Emotion.ANGER,
    Emotion.DISGUST,
    Emotion.FEAR,
    Emotion.HAPPINESS,
    Emotion.SADNESS,
    Emotion.SURPRISE,
    Emotion.NEUTRAL,
]

# Okabe-Ito colorblind-safe palette; neutral rendered as grey.
EMOTION_COLORS: dict[Emotion, str] = {
    Emotion.ANGER: "#D55E00",      # vermillion
    Emotion.DISGUST: "#009E73",    # bluish green
    Emotion.FEAR: "#CC79A7",       # reddish purple
    Emotion.HAPPINESS: "#E69F00",  # orange
    Emotion.SADNESS: "#0072B2",    # blue
    Emotion.SURPRISE: "#F0E442",   # yellow
    Emotion.NEUTRAL: "#999999",    # grey
}
