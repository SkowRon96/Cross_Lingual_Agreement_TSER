"""Model registry: model_id -> concrete EmotionModel factory (fixed roster)."""
from collections.abc import Callable

from app.config.settings import Settings
from app.models.base import EmotionModel
from app.models.text.hf_text import HFTextModel

_settings = Settings()

_FACTORIES: dict[str, Callable[[], EmotionModel]] = {
    "en_text_hartmann": lambda: HFTextModel(
        "en_text_hartmann",
        "j-hartmann/emotion-english-distilroberta-base",
        cache_dir=_settings.hf_home,
    ),
}

MODEL_IDS: list[str] = list(_FACTORIES)


def build_model(model_id: str) -> EmotionModel:
    if model_id not in _FACTORIES:
        raise KeyError(f"Unknown model_id: {model_id!r}. Known: {MODEL_IDS}")
    return _FACTORIES[model_id]()
