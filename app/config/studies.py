"""Static study design: which models ran where, headline vs secondary/leaky, and the
canonical classes each model can emit (drives the §4 common-support rule)."""
from app.config.emotions import Emotion

MODALITY: dict[str, str] = {
    "en_text_hartmann": "text", "xlmr_pooled10": "text", "mling_tabularis": "text",
    "pl_text_roberta6": "text", "pl_text_lstm": "text",
    "pl_text_roberta_audit_s7": "text", "pl_text_roberta_audit_s17": "text",
    "pl_text_roberta_audit_s27": "text",
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
    "pl_text_roberta_audit_s7", "pl_text_roberta_audit_s17", "pl_text_roberta_audit_s27",
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
    # pl_text_roberta_audit_s{7,17,27}: tf-polish-roberta-large-v2, trained in a separate
    # annotation-audit project and imported here (not run by this code). The p_0..p_6 label
    # order was CONFIRMED against that project's own corpus, not assumed -> full 7-class
    # support. Three seeds, reported as mean +- sd.
    "pl_text_roberta_audit_s7": set(_ALL7),
    "pl_text_roberta_audit_s17": set(_ALL7),
    "pl_text_roberta_audit_s27": set(_ALL7),
}

ROSTER: dict[tuple[int, str], list[str]] = {
    (1, "nEMO"): ["mling_tabularis", "pl_text_roberta6", "xlmr_pooled10", "pl_text_lstm",
                  "speech_zeroshot", "speech_cnn_tf_nemo",
                  "pl_text_roberta_audit_s7", "pl_text_roberta_audit_s17",
                  "pl_text_roberta_audit_s27"],
    (1, "RAVDESS"): ["en_text_hartmann", "mling_tabularis", "speech_zeroshot",
                     "speech_cnn_tf_ravdess", "en_speech_iemocap", "en_face_vit"],
    # MELD (real-dialogue EN, Study 1): natural emotional text -> measurable text<->speech
    # agreement + acted-vs-natural contrast with RAVDESS.
    # speech_cnn_tf_ravdess on MELD is CROSS-dataset (not leaky) -> a generalization probe.
    # en_face_vit on MELD = LARGEST-FACE best-effort: multi-speaker frames / cutaways mean the
    # detected face may not be the utterance speaker -> NOISY, SECONDARY, never headline.
    (1, "MELD"): ["en_text_hartmann", "mling_tabularis", "speech_zeroshot",
                  "en_speech_iemocap", "speech_cnn_tf_ravdess", "en_face_vit"],
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
