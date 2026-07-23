"""Ingest orchestrator: resolve gold labels, write the index, report harmonization."""
from pathlib import Path

from pydantic import BaseModel

from app.data.index_io import write_index
from app.data.schema import Sample
from app.labels.mapping import map_label


class IngestReport(BaseModel):
    corpus: str
    total: int
    kept: int
    dropped: int
    per_raw_label: dict[str, str]     # raw label -> canonical value or "DROP"
    dropped_labels: dict[str, int]    # raw label -> count of dropped samples


def resolve_gold(samples: list[Sample]) -> list[Sample]:
    resolved = []
    for s in samples:
        gold = map_label(s.gold_raw)  # raises KeyError on unknown label (fail loud)
        resolved.append(s.model_copy(update={"gold": gold}))
    return resolved


def ingest_corpus(samples: list[Sample], corpus: str, out_dir: Path) -> IngestReport:
    resolved = resolve_gold(samples)
    per_raw_label: dict[str, str] = {}
    dropped_labels: dict[str, int] = {}
    kept = 0
    for s in resolved:
        canonical = s.gold.value if s.gold is not None else "DROP"
        per_raw_label[s.gold_raw] = canonical
        if s.gold is None:
            dropped_labels[s.gold_raw] = dropped_labels.get(s.gold_raw, 0) + 1
        else:
            kept += 1
    write_index(resolved, Path(out_dir) / f"{corpus}.parquet")
    return IngestReport(
        corpus=corpus,
        total=len(resolved),
        kept=kept,
        dropped=len(resolved) - kept,
        per_raw_label=per_raw_label,
        dropped_labels=dropped_labels,
    )
