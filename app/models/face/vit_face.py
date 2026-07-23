"""ViT facial-expression emotion model over extracted video face crops."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.data.schema import Sample
from app.data.video import extract_face_frames
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction
from app.models.text.hf_text import native_labels_from_id2label


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def aggregate_face_scores(per_crop_probs: list[list[float]]) -> list[float]:
    return np.asarray(per_crop_probs, dtype=np.float64).mean(axis=0).tolist()


class ViTFaceModel:
    modality = "face"

    def __init__(self, model_id: str, hf_repo: str, cache_dir: Path | None = None,
                 num_frames: int = 8):
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.num_frames = num_frames
        cd = str(cache_dir) if cache_dir else None
        self.processor = AutoImageProcessor.from_pretrained(hf_repo, cache_dir=cd)
        self.model = AutoModelForImageClassification.from_pretrained(hf_repo, cache_dir=cd)
        self.device = _device()
        self.model.to(self.device).eval()
        self.native_labels = native_labels_from_id2label(self.model.config.id2label)
        self.mapper = LabelMapper(self.native_labels)

    def config_signature(self) -> str:
        return f"{self.hf_repo}@nframes={self.num_frames}"

    def _abstain(self, sample_id: str, reason: str) -> Prediction:
        return Prediction(sample_id=sample_id, model_id=self.model_id, modality=self.modality,
                          pred_raw=None, pred=None, confidence=None, scores={},
                          abstained=True, abstain_reason=reason)

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        out: list[Prediction] = []
        for s in samples:
            if s.video_path is None:
                out.append(self._abstain(s.sample_id, "no video_path on sample"))
                continue
            crops = extract_face_frames(s.video_path, num_frames=self.num_frames)
            if not crops:
                out.append(self._abstain(s.sample_id, "no face detected in any sampled frame"))
                continue
            inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (n_crops, n_labels)
            mean_scores = aggregate_face_scores(probs.tolist())
            out.append(to_prediction(s.sample_id, self.model_id, self.modality,
                                     self.native_labels, mean_scores, self.mapper))
        return out
