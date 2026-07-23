"""Master raw-label -> canonical-emotion mapping (single source of truth).

Applied identically to gold labels and model predictions. Unknown labels
raise KeyError so a new/unexpected label can never be silently mishandled.
"""
from app.config.emotions import Emotion

# Raw label string (lowercased) -> canonical Emotion.
MASTER_MAP: dict[str, Emotion] = {
    "anger": Emotion.ANGER,
    "angry": Emotion.ANGER,
    "disgust": Emotion.DISGUST,
    "fear": Emotion.FEAR,
    "happiness": Emotion.HAPPINESS,
    "happy": Emotion.HAPPINESS,
    "joy": Emotion.HAPPINESS,
    "sadness": Emotion.SADNESS,
    "sad": Emotion.SADNESS,
    "surprise": Emotion.SURPRISE,
    "surprised": Emotion.SURPRISE,
    "neutral": Emotion.NEUTRAL,
    # Established merges (see spec section 4).
    "calm": Emotion.NEUTRAL,
    "excitement": Emotion.HAPPINESS,
    "enthusiasm": Emotion.HAPPINESS,
}

# Labels intentionally dropped (non-Ekman, no established equivalence).
# "none of them" is dropped by default; models lacking an explicit neutral
# override it to Emotion.NEUTRAL at LabelMapper construction.
DROP_LABELS: set[str] = {
    "hope", "pride", "contempt", "frustration", "gratitude", "love",
    "sarcasm", "anxiety", "apology", "assertiveness", "concern",
    "encouragement", "poker", "none of them",
}


def map_label(
    raw: str,
    overrides: dict[str, Emotion | None] | None = None,
) -> Emotion | None:
    """Map a raw label to a canonical Emotion, or None if dropped.

    Resolution order: per-model overrides -> master map -> drop set ->
    KeyError. `overrides` keys are matched after the same normalization.
    """
    key = raw.strip().lower()
    if overrides:
        norm_overrides = {k.strip().lower(): v for k, v in overrides.items()}
        if key in norm_overrides:
            return norm_overrides[key]
    if key in MASTER_MAP:
        return MASTER_MAP[key]
    if key in DROP_LABELS:
        return None
    raise KeyError(f"Unknown emotion label: {raw!r}")
