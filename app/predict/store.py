"""PredictionStore: cached, versioned parquet store for model predictions."""
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from app.config.emotions import Emotion
from app.models.base import Prediction


class PredictionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    # ----- paths & signatures -----
    def path(self, study: int, corpus: str, model_id: str) -> Path:
        return self.root / f"study{study}" / corpus / f"{model_id}.parquet"

    def signature(self, model_id: str, config_hash: str, sample_ids: Iterable[str]) -> str:
        ids_digest = hashlib.sha256(
            "\n".join(sorted(sample_ids)).encode("utf-8")
        ).hexdigest()
        payload = f"{model_id}|{config_hash}|{ids_digest}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ----- serialization helpers -----
    @staticmethod
    def _to_row(p: Prediction) -> dict:
        d = p.model_dump(mode="json")
        d["scores"] = json.dumps(d["scores"])  # dict -> JSON string column
        return d

    @staticmethod
    def _from_row(row: dict) -> Prediction:
        scores_raw = json.loads(row["scores"]) if row["scores"] else {}
        scores = {Emotion(k): float(v) for k, v in scores_raw.items()}
        return Prediction(
            sample_id=row["sample_id"], model_id=row["model_id"], modality=row["modality"],
            pred_raw=row["pred_raw"], pred=(Emotion(row["pred"]) if row["pred"] else None),
            confidence=(None if pd.isna(row["confidence"]) else float(row["confidence"])),
            scores=scores, abstained=bool(row["abstained"]),
            abstain_reason=(row["abstain_reason"] or None),
        )

    # ----- read / write -----
    def write(self, study: int, corpus: str, model_id: str,
              predictions: list[Prediction], signature: str) -> Path:
        out = self.path(study, corpus, model_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([self._to_row(p) for p in predictions])
        df.to_parquet(out, index=False)
        self._record_manifest(study, corpus, model_id, signature, len(predictions))
        return out

    def read(self, study: int, corpus: str, model_id: str) -> list[Prediction]:
        df = pd.read_parquet(self.path(study, corpus, model_id))
        return [self._from_row(row) for row in df.to_dict(orient="records")]

    def existing_sample_ids(self, study: int, corpus: str, model_id: str) -> set[str]:
        p = self.path(study, corpus, model_id)
        if not p.exists():
            return set()
        return set(pd.read_parquet(p, columns=["sample_id"])["sample_id"].tolist())

    def append(self, study: int, corpus: str, model_id: str,
               predictions: list[Prediction]) -> None:
        p = self.path(study, corpus, model_id)
        new = pd.DataFrame([self._to_row(x) for x in predictions])
        if p.exists():
            existing = pd.read_parquet(p)
            new = pd.concat([existing, new], ignore_index=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(p, index=False)

    # ----- manifest / cache -----
    def _record_manifest(self, study: int, corpus: str, model_id: str,
                          signature: str, n: int) -> None:
        manifest = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        key = f"study{study}/{corpus}/{model_id}"
        manifest[key] = {"signature": signature, "n_samples": n}
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def is_cached(self, study: int, corpus: str, model_id: str, signature: str) -> bool:
        if not self.path(study, corpus, model_id).exists():
            return False
        if not self.manifest_path.exists():
            return False
        manifest = json.loads(self.manifest_path.read_text())
        key = f"study{study}/{corpus}/{model_id}"
        return manifest.get(key, {}).get("signature") == signature
