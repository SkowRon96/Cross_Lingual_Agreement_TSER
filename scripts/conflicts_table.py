"""Cross-modal conflict table: every model pair, all Study-1 corpora.

For each corpus (nEMO, RAVDESS, MELD) and every pair of models with stored predictions,
count the samples where the two disagree (both non-empty) and split them by which model
is closer to the true emotion. Emits results/tables/article/conflicts.{tex,md,csv}.

Reuses app.analysis.conflicts.conflict (the same logic behind the bundle's headline pair),
so numbers match the analysis exactly — just extended to all pairs.

  PYTHONPATH=. .venv/bin/python scripts/conflicts_table.py
"""
from itertools import combinations
from pathlib import Path

import pandas as pd

from app.analysis.conflicts import conflict
from app.analysis.run import _load, gold_lookup, _index_path
from app.config import studies
from app.config.settings import Settings
from app.predict.store import PredictionStore

s = Settings()
OUT = s.results_root / "tables" / "article"
OUT.mkdir(parents=True, exist_ok=True)
store = PredictionStore(s.results_root / "predictions")
index_dir = Path(s.index_dir)

CORPORA = ["nEMO", "RAVDESS", "MELD"]


def _tag(mid: str) -> str:
    """model id annotated with its modality, e.g. 'en_text_hartmann (text)'."""
    return f"{mid} ({studies.MODALITY[mid]})"


rows = []
for corpus in CORPORA:
    if not _index_path(index_dir, 1, corpus).exists():
        continue
    gold = gold_lookup(1, corpus, index_dir)
    present = {}
    for mid in studies.ROSTER[(1, corpus)]:
        preds = _load(store, 1, corpus, mid)
        if preds is not None:
            present[mid] = preds
    for a, b_ in combinations(present, 2):
        c = conflict(present[a], present[b_], gold, a, b_, corpus)
        if c.n_conflicts == 0:
            continue
        rows.append({
            "corpus": corpus,
            "model A": _tag(a),
            "model B": _tag(b_),
            "n conflicts": c.n_conflicts,
            "A closer to true": c.a_closer_to_gold,
            "B closer to true": c.b_closer_to_gold,
            "both wrong": c.both_wrong,
        })

df = pd.DataFrame(rows)
(OUT / "conflicts.md").write_text(df.to_markdown(index=False))
(OUT / "conflicts.tex").write_text(df.to_latex(
    index=False,
    caption=("Cross-modal conflicts (Study 1): for every model pair and corpus, "
             "number of samples where the two models disagree, split by which model "
             "is closer to the true emotion."),
    escape=True))

print(df.to_markdown(index=False))
print(f"\n-> {OUT}/conflicts.{{tex,md}}")
