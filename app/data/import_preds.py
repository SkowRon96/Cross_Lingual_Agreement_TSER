"""Import externally-computed predictions (e.g. Qwen2-Audio from the Windows GPU box)."""
from pathlib import Path

import pandas as pd

from app.config.emotions import Emotion
from app.models.base import Prediction
from app.predict.store import PredictionStore


def import_predictions(preds_path: Path, store: PredictionStore, study: int, corpus: str,
                       model_id: str = "speech_zeroshot",
                       config_hash: str = "qwen2-audio-7b-instruct@zeroshot") -> int:
    df = pd.read_parquet(preds_path)
    rows = [r for r in df.to_dict(orient="records") if r["corpus"] == corpus]
    predictions = []
    for r in rows:
        pred_str = (r.get("pred") or "").strip()
        if pred_str:
            emo = Emotion(pred_str)  # raises ValueError on a non-canonical value (fail loud)
            predictions.append(Prediction(
                sample_id=r["sample_id"], model_id=model_id, modality="speech",
                pred_raw=str(r.get("raw_response") or ""), pred=emo, confidence=None,
                scores={emo: 1.0}, abstained=False, abstain_reason=None))
        else:
            predictions.append(Prediction(
                sample_id=r["sample_id"], model_id=model_id, modality="speech",
                pred_raw=str(r.get("raw_response") or ""), pred=None, confidence=None,
                scores={}, abstained=True, abstain_reason="zero-shot response unparseable"))
    # write() takes the plain config_hash and derives the signature itself.
    store.write(study, corpus, model_id, predictions, config_hash)
    return len(predictions)
