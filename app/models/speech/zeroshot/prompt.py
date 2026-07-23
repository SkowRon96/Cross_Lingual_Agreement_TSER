"""Canonical 7-class prompt + robust response parser for the Qwen2-Audio zero-shot model.

This module is the SOURCE OF TRUTH for the prompt and the answer parser. The standalone
Windows inference script (scripts/qwen_windows/run_qwen.py) copies this logic verbatim;
tests here guard the canonical version.
"""
CANONICAL_WORDS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# Synonym / surface-form -> canonical word. Ordered longest-first at match time.
_SYNONYMS = {
    "anger": "anger", "angry": "anger", "annoyed": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear", "afraid": "fear", "scared": "fear",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness", "joyful": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral", "calm": "neutral", "none": "neutral",
}


def build_prompt() -> str:
    words = ", ".join(CANONICAL_WORDS)
    return ("Listen to the speech and classify the speaker's emotion into exactly one of "
            f"these categories: {words}. Answer with only one word from that list.")


def parse_response(raw: str) -> str | None:
    if not raw:
        return None
    text = raw.strip().lower()
    # scan tokens; longest synonym keys first so 'surprised' beats a substring match
    for key in sorted(_SYNONYMS, key=len, reverse=True):
        # word-ish containment: the key appears as a standalone token or substring of a word
        if key in text:
            return _SYNONYMS[key]
    return None
