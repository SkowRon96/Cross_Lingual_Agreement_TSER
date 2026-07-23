"""Static study design: which models ran where, headline vs secondary/leaky, and the
canonical classes each model can emit (drives the §4 common-support rule)."""
from app.config.emotions import Emotion

MODALITY: dict[str, str] = {
    "en_text_hartmann": "text", "xlmr_pooled10": "text", "mling_tabularis": "text",
    "pl_text_roberta6": "text", "pl_text_lstm": "text",
    "speech_zeroshot": "speech", "speech_cnn_tf_ravdess": "speech",
    "speech_cnn_tf_nemo": "speech", "en_speech_iemocap": "speech",
    "en_face_vit": "face",
}

HEADLINE: dict[tuple[str, str], str] = {
    ("PL", "text"): "mling_tabularis", ("PL", "speech"): "speech_zeroshot",
    ("EN", "text"): "en_text_hartmann", ("EN", "speech"): "speech_zeroshot",
}

SECONDARY: set[str] = {
    "pl_text_roberta6", "xlmr_pooled10", "pl_text_lstm",
    "speech_cnn_tf_ravdess", "speech_cnn_tf_nemo", "en_speech_iemocap",
}
LEAKY: set[str] = {"pl_text_lstm", "speech_cnn_tf_ravdess", "speech_cnn_tf_nemo"}

_ALL7 = set(Emotion)
_NO_SURPRISE = _ALL7 - {Emotion.SURPRISE}
SUPPORTED: dict[str, set[Emotion]] = {
    "en_text_hartmann": set(_ALL7), "mling_tabularis": set(_ALL7),
    "pl_text_lstm": set(_ALL7), "speech_zeroshot": set(_ALL7),
    "speech_cnn_tf_ravdess": set(_ALL7), "speech_cnn_tf_nemo": set(_ALL7),
    "en_face_vit": set(_ALL7),
    "xlmr_pooled10": set(_NO_SURPRISE),
    "pl_text_roberta6": set(_NO_SURPRISE),  # assumed label order; no surprise
    "en_speech_iemocap": {Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL},
}

ROSTER: dict[tuple[int, str], list[str]] = {
    (1, "nEMO"): ["mling_tabularis", "pl_text_roberta6", "xlmr_pooled10", "pl_text_lstm",
                  "speech_zeroshot", "speech_cnn_tf_nemo"],
    (1, "RAVDESS"): ["en_text_hartmann", "mling_tabularis", "speech_zeroshot",
                     "speech_cnn_tf_ravdess", "en_speech_iemocap", "en_face_vit"],
}
STUDY2_SPLITS: list[str] = ["cafe", "crema_d", "emns", "emozionalmente", "enterface",
                            "jl_corpus", "mesd", "nemo", "oreau", "pavoque", "ravdess",
                            "resd", "subesco"]
for _s in STUDY2_SPLITS:
    ROSTER[(2, _s)] = ["mling_tabularis", "speech_zeroshot"]

# CAMEO split -> ISO language (source language of each sub-corpus).
SPLIT_LANG: dict[str, str] = {
    "cafe": "fr", "crema_d": "en", "emns": "en", "emozionalmente": "it",
    "enterface": "en", "jl_corpus": "en", "mesd": "es", "nemo": "pl",
    "oreau": "fr", "pavoque": "de", "ravdess": "en", "resd": "ru", "subesco": "bn",
}
