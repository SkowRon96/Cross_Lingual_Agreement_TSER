"""Generic Hugging Face sequence-classification text emotion model."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction


def native_labels_from_id2label(id2label: dict) -> list[str]:
    return [id2label[k] for k in sorted(id2label, key=lambda x: int(x))]


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class HFTextModel:
    modality = "text"

    def __init__(self, model_id: str, hf_repo: str, cache_dir: Path | None = None):
        self.model_id = model_id
        self.hf_repo = hf_repo
        cd = str(cache_dir) if cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, cache_dir=cd)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_repo, cache_dir=cd)
        self.device = _device()
        self.model.to(self.device).eval()
        self.native_labels = native_labels_from_id2label(self.model.config.id2label)
        self.mapper = LabelMapper(self.native_labels)

    def config_signature(self) -> str:
        return self.hf_repo

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        texts = [s.text for s in samples]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        return [
            to_prediction(s.sample_id, self.model_id, self.modality,
                          self.native_labels, probs[i].tolist(), self.mapper)
            for i, s in enumerate(samples)
        ]
