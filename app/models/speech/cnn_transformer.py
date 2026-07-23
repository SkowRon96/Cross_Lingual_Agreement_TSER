"""Custom CNN-Transformer speech emotion model (own model; leaky on RAVDESS/nEMO).

StandardScaler is refit per-corpus (the original training scaler was not persisted
and the train split was unseeded, so it is unrecoverable) — an explicit approximation.

CALLERS MUST pass the whole corpus in one predict_batch() call. The scaler is fit
fresh on whatever batch arrives, so a partial-corpus batch silently degrades results,
and a batch of fewer than 2 samples is degenerate (zero variance -> all-zero features).
"""
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from app.data.audio import load_audio
from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction
from app.models.speech.architectures import ParallelModel
from app.models.speech.features import mel_spectrogram, to_model_signal


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class CNNTransformerSpeechModel:
    modality = "speech"

    def __init__(self, model_id: str, weights_path: Path, native_labels: list[str]):
        self.model_id = model_id
        self.weights_path = Path(weights_path)
        self.native_labels = native_labels
        self.device = _device()
        self.model = ParallelModel(len(native_labels))
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.mapper = LabelMapper(native_labels)

    def config_signature(self) -> str:
        return f"cnn_tf:{self.weights_path.name}@labels={','.join(self.native_labels)}@scaler=per-corpus"

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        if len(samples) < 2:
            warnings.warn(
                f"{self.model_id}: StandardScaler is refit per batch; a batch of "
                f"{len(samples)} sample(s) yields degenerate/inconsistent features. "
                "Run this model with the whole corpus in one batch.", stacklevel=2)
        mels = []
        for s in samples:
            audio, sr = load_audio(s)
            mels.append(mel_spectrogram(to_model_signal(audio, sr)))
        X = np.stack(mels)[:, None, :, :]                      # (n,1,128,563)
        b, c, h, w = X.shape
        Xf = StandardScaler().fit_transform(X.reshape(b, -1))  # per-corpus refit
        X = Xf.reshape(b, c, h, w).astype("float32")
        with torch.no_grad():
            _, soft = self.model(torch.from_numpy(X).to(self.device))
        probs = soft.cpu().numpy()
        return [
            to_prediction(s.sample_id, self.model_id, self.modality,
                          self.native_labels, probs[i].tolist(), self.mapper)
            for i, s in enumerate(samples)
        ]
