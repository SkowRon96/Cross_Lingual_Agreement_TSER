"""Model contract: Prediction, EmotionModel protocol, and to_prediction() builder."""
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.config.emotions import Emotion
from app.data.schema import Sample
from app.labels.mapper import LabelMapper


class Prediction(BaseModel):
    sample_id: str
    model_id: str
    modality: str
    pred_raw: str | None
    pred: Emotion | None
    confidence: float | None
    scores: dict[Emotion, float]
    abstained: bool = False
    abstain_reason: str | None = None


@runtime_checkable
class EmotionModel(Protocol):
    model_id: str
    modality: str

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]: ...


def to_prediction(
    sample_id: str,
    model_id: str,
    modality: str,
    native_labels: Sequence[str],
    scores: Sequence[float],
    mapper: LabelMapper,
) -> Prediction:
    """Build a Prediction from a model's native scores via the mapper.

    Records the native argmax as pred_raw, maps+aggregates into canonical
    scores, then takes the canonical argmax as pred and its value as
    confidence. Abstains (pred=None) when all mass maps to dropped labels.
    """
    native_argmax = native_labels[max(range(len(scores)), key=lambda i: scores[i])]
    canonical = mapper.map_scores(scores)
    if not canonical:
        return Prediction(
            sample_id=sample_id, model_id=model_id, modality=modality,
            pred_raw=native_argmax, pred=None, confidence=None, scores={},
            abstained=True,
            abstain_reason="all probability mass mapped to dropped labels",
        )
    pred = max(canonical, key=canonical.get)
    return Prediction(
        sample_id=sample_id, model_id=model_id, modality=modality,
        pred_raw=native_argmax, pred=pred, confidence=canonical[pred],
        scores=canonical, abstained=False, abstain_reason=None,
    )
