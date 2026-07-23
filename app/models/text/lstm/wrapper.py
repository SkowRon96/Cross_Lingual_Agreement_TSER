"""LSTMTextModel: EmotionModel that runs the isolated Keras inference subprocess."""
import json
import subprocess
from pathlib import Path

from app.data.schema import Sample
from app.labels.mapper import LabelMapper
from app.models.base import Prediction, to_prediction

NATIVE_LABELS = ["neutral", "happiness", "sadness", "fear", "anger", "disgust", "surprise"]


class LSTMTextModel:
    modality = "text"

    def __init__(self, model_id: str, model_dir: Path, tf_python: Path, infer_script: Path):
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        self.tf_python = Path(tf_python)
        self.infer_script = Path(infer_script)
        self.native_labels = NATIVE_LABELS
        self.mapper = LabelMapper(NATIVE_LABELS)

    def config_signature(self) -> str:
        return f"lstm:{self.model_dir.name}"

    def predict_batch(self, samples: list[Sample]) -> list[Prediction]:
        texts = [s.text for s in samples]
        proc = subprocess.run(
            [str(self.tf_python), str(self.infer_script), str(self.model_dir)],
            input=json.dumps({"texts": texts}), text=True, capture_output=True,
            env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin"},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LSTM inference subprocess failed:\n{proc.stderr[-800:]}")
        out = json.loads(proc.stdout)
        if out["labels"] != self.native_labels:
            raise ValueError(f"Unexpected label order from infer.py: {out['labels']}")
        probs = out["probs"]
        return [
            to_prediction(s.sample_id, self.model_id, self.modality,
                          self.native_labels, probs[i], self.mapper)
            for i, s in enumerate(samples)
        ]
