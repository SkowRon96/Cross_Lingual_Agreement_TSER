"""Generic Hugging Face sequence-classification text emotion model."""
from pathlib import Path

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

    def __init__(self, model_id: str, hf_repo: str, cache_dir: Path | None = None,
                 id2label_override: dict | None = None,
                 label_overrides: dict | None = None, tokenizer_repo: str | None = None):
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.tokenizer_repo = tokenizer_repo
        cd = str(cache_dir) if cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo or hf_repo, cache_dir=cd)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_repo, cache_dir=cd)
        self.device = _device()
        self.model.to(self.device).eval()
        self._overridden = id2label_override is not None
        if id2label_override is not None:
            n = self.model.config.num_labels
            if len(id2label_override) != n:
                raise ValueError(
                    f"id2label_override has {len(id2label_override)} entries but model has {n} labels")
            self.native_labels = native_labels_from_id2label(id2label_override)
        else:
            self.native_labels = native_labels_from_id2label(self.model.config.id2label)
        self.label_overrides = label_overrides
        self.mapper = LabelMapper(self.native_labels, overrides=label_overrides)

    def config_signature(self) -> str:
        sig = self.hf_repo
        if self.tokenizer_repo:
            sig += f"@tok={self.tokenizer_repo}"
        if self._overridden:
            sig += "@labels=" + ",".join(self.native_labels)
        if self.label_overrides:
            import json
            items = sorted((k, (v.value if v is not None else "DROP"))
                           for k, v in self.label_overrides.items())
            sig += "@ovr=" + json.dumps(items)
        return sig

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        texts = [s.text for s in samples]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return [
            to_prediction(s.sample_id, self.model_id, self.modality,
                          self.native_labels, probs[i].tolist(), self.mapper)
            for i, s in enumerate(samples)
        ]
