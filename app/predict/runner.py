"""Prediction runner: cache-aware, resumable batched inference into the store."""
from collections.abc import Iterable, Iterator

from app.data.schema import Sample
from app.models.base import EmotionModel
from app.predict.store import PredictionStore


def chunked(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run_prediction(samples: list[Sample], model: EmotionModel, store: PredictionStore,
                   study: int, corpus: str, *, force: bool = False,
                   batch_size: int = 32) -> str:
    cfg = model.config_signature()
    target_ids = [s.sample_id for s in samples]
    if not force and store.is_cached(study, corpus, model.model_id, cfg, target_ids):
        return "cached"
    done = set() if force else store.existing_sample_ids(study, corpus, model.model_id)
    todo = [s for s in samples if s.sample_id not in done]
    first = not done  # force or empty store -> the first write (re)creates the file
    for batch in chunked(todo, batch_size):
        preds = model.predict_batch(batch)
        if first:
            store.write(study, corpus, model.model_id, preds, cfg)
            first = False
        else:
            store.append(study, corpus, model.model_id, preds)
    return "done"
