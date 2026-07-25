"""Model registry: model_id -> concrete EmotionModel factory (fixed roster)."""
from collections.abc import Callable

from app.config.emotions import Emotion
from app.config.settings import Settings
from app.models.base import EmotionModel
from app.models.face.vit_face import ViTFaceModel
from app.models.speech.cnn_transformer import CNNTransformerSpeechModel
from app.models.speech.speechbrain_iemocap import SpeechBrainIEMOCAP
from app.models.text.hf_text import HFTextModel
from app.models.text.lstm.wrapper import LSTMTextModel

_settings = Settings()

# roberta6: repo config is generic (LABEL_0..5); this label order is an ASSUMPTION
# (unverified — no label2id published; README fingerprint / nEMO probe inconclusive).
# roberta6 is a SECONDARY model, never used in a headline result. See spec §4/§7.
ROBERTA6_ID2LABEL = {0: "anger", 1: "fear", 2: "disgust", 3: "sadness", 4: "joy", 5: "none of them"}

RAVDESS8_LABELS = ["surprise", "neutral", "calm", "happy", "sad", "angry", "fear", "disgust"]
NEMO6_LABELS = ["anger", "fear", "happiness", "sadness", "surprised", "neutral"]

_FACTORIES: dict[str, Callable[[], EmotionModel]] = {
    "en_text_hartmann": lambda: HFTextModel(
        "en_text_hartmann",
        "j-hartmann/emotion-english-distilroberta-base",
        cache_dir=_settings.hf_home,
    ),
    "xlmr_pooled10": lambda: HFTextModel(
        "xlmr_pooled10", "poltextlab/xlm-roberta-large-pooled-emotions10-v3",
        cache_dir=_settings.hf_home, tokenizer_repo="xlm-roberta-large"),
    "mling_tabularis": lambda: HFTextModel(
        "mling_tabularis", "tabularisai/multilingual-emotion-classification",
        cache_dir=_settings.hf_home),
    "pl_text_roberta6": lambda: HFTextModel(
        "pl_text_roberta6", "visegradmedia-emotion/Emotion_RoBERTa_polish6",
        cache_dir=_settings.hf_home,
        id2label_override=ROBERTA6_ID2LABEL,
        label_overrides={"none of them": Emotion.NEUTRAL}),
    "en_face_vit": lambda: ViTFaceModel(
        "en_face_vit", "trpakov/vit-face-expression", cache_dir=_settings.hf_home),
    # _v2 = retrained from the corrected notebooks (augmentation-label bug fixed, NumPy-2
    # scaler fix, seeded split, early stopping). Honest held-out: RAVDESS 0.713 (vs the
    # 2022 file's 0.613), nEMO 0.887 (vs the broken _v1's 0.287). Both are LEAKY/secondary
    # models (trained on RAVDESS/nEMO); the clean/headline speech model is speech_zeroshot.
    "speech_cnn_tf_ravdess": lambda: CNNTransformerSpeechModel(
        "speech_cnn_tf_ravdess",
        _settings.speech_weights_dir / "cnn_transf_parallel_model_v2.pt", RAVDESS8_LABELS),
    "speech_cnn_tf_nemo": lambda: CNNTransformerSpeechModel(
        "speech_cnn_tf_nemo",
        _settings.speech_weights_dir / "cnn_transf_parallel_model_nemo_v2.pt", NEMO6_LABELS),
    "en_speech_iemocap": lambda: SpeechBrainIEMOCAP("en_speech_iemocap", cache_dir=_settings.hf_home),
    "pl_text_lstm": lambda: LSTMTextModel(
        "pl_text_lstm", _settings.lstm_model_dir, _settings.tf_python, _settings.lstm_infer_script),
}

MODEL_IDS: list[str] = list(_FACTORIES)


def build_model(model_id: str) -> EmotionModel:
    if model_id not in _FACTORIES:
        raise KeyError(f"Unknown model_id: {model_id!r}. Known: {MODEL_IDS}")
    return _FACTORIES[model_id]()
