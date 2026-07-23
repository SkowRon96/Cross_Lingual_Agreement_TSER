"""speechbrain wav2vec2 IEMOCAP speech-emotion model (clean EN reference; 4 classes)."""
from pathlib import Path

import numpy as np
import torch

from app.data.audio import load_audio
from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction

# IEMOCAP label abbreviations -> canonical raw words (mapped further by map_label).
IEMOCAP_TO_CANONICAL = {"ang": "anger", "hap": "happiness", "sad": "sadness", "neu": "neutral"}
_TARGET_SR = 16000


class SpeechBrainIEMOCAP:
    modality = "speech"

    def __init__(self, model_id: str, cache_dir: Path | None = None):
        from speechbrain.inference.interfaces import foreign_class
        self.model_id = model_id
        self.classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=str(cache_dir) if cache_dir else None,
        )
        # Label order from the model's encoder, mapped to canonical raw words.
        enc = self.classifier.hparams.label_encoder
        self.native_labels = [
            IEMOCAP_TO_CANONICAL[enc.decode_ndim(i)] for i in range(len(IEMOCAP_TO_CANONICAL))
        ]
        self.mapper = LabelMapper(self.native_labels)

    def config_signature(self) -> str:
        return "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        out = []
        for s in samples:
            audio, sr = load_audio(s)
            wav = torch.from_numpy(np.asarray(audio, dtype="float32"))
            if sr != _TARGET_SR:
                import torchaudio
                wav = torchaudio.functional.resample(wav, sr, _TARGET_SR)
            out_prob, _, _, _ = self.classifier.classify_batch(wav.unsqueeze(0))
            probs = out_prob.squeeze(0).exp().cpu().numpy()  # log-probs -> probs
            probs = (probs / probs.sum()).astype("float64")
            out.append(to_prediction(s.sample_id, self.model_id, self.modality,
                                     self.native_labels, probs.tolist(), self.mapper))
        return out
