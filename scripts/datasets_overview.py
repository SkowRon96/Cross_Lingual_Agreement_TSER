"""Compile a dataset-overview table from the ingested indices + harmonization reports.

Emits results/tables/datasets_overview.{md,csv} and a per-split CAMEO breakdown. Numbers
come from results/index/*/*.report.json (authoritative kept/dropped + native label sets),
languages/modalities from app.config.studies, descriptions hand-curated below.

  PYTHONPATH=. .venv/bin/python scripts/datasets_overview.py
"""
import csv
import json
from pathlib import Path

from app.config import studies
from app.config.settings import Settings

s = Settings()
IDX = s.index_dir

# hand-curated characteristic notes (accurate to how each corpus behaves in this study)
NOTES = {
    "nEMO": "Acted Polish speech, 9 professional actors; read sentences are semantically ~neutral "
            "→ weak text-channel emotion. Own dataset (HF).",
    "RAVDESS": "Acted English, 24 actors; trimodal (Zenodo). Only 2 fixed neutral sentences "
               "(\"Kids are talking by the door\" / \"Dogs are sitting by the door\") → emotionally "
               "void text channel. Includes 'calm' (→ neutral).",
    "MELD": "Real multi-party dialogue from the TV series Friends; natural emotional text; trimodal "
            "(video). Heavy neutral skew (~47%); noisy audio (laugh track, overlapping speech).",
    "CAMEO": "Multilingual benchmark aggregating 13 sub-corpora across 8 languages; acted/read "
             "speech; heterogeneous native labels → non-Ekman categories (sarcasm, poker, anxiety…) "
             "dropped in harmonization.",
}
MODALITIES = {"nEMO": "text + speech", "RAVDESS": "text + speech + face",
              "MELD": "text + speech + face*", "CAMEO": "text + speech"}
LANG = {"nEMO": "pl", "RAVDESS": "en", "MELD": "en"}


def read_report(study, corpus):
    p = IDX / f"study{study}" / f"{corpus}.report.json"
    return json.loads(p.read_text()) if p.exists() else None


def native_and_dropped(rep):
    kept = sorted({v for v in rep["per_raw_label"].values() if v != "DROP"})
    native = sorted(rep["per_raw_label"].keys())
    dropped = sorted(rep.get("dropped_labels", {}).keys())
    return native, kept, dropped


rows = []
# --- Study 1 datasets ---
for corpus, study in [("nEMO", 1), ("RAVDESS", 1), ("MELD", 1)]:
    rep = read_report(study, corpus)
    if not rep:
        continue
    native, kept, dropped = native_and_dropped(rep)
    rows.append({
        "dataset": corpus, "study": study, "language": LANG[corpus],
        "modalities": MODALITIES[corpus],
        "native_emotions": ", ".join(native),
        "canonical_emotions": ", ".join(kept),
        "dropped": ", ".join(dropped) or "—",
        "n_kept": rep["kept"], "n_total": rep["total"],
        "notes": NOTES[corpus],
    })

# --- CAMEO (Study 2): aggregate across 13 splits ---
cameo_langs, cameo_native, cameo_kept, cameo_dropped = set(), set(), set(), set()
cameo_total = cameo_keptn = 0
cameo_rows = []
for split in studies.STUDY2_SPLITS:
    rep = read_report(2, split)
    if not rep:
        continue
    native, kept, dropped = native_and_dropped(rep)
    iso = studies.SPLIT_LANG[split]
    cameo_langs.add(iso); cameo_native |= set(native); cameo_kept |= set(kept); cameo_dropped |= set(dropped)
    cameo_total += rep["total"]; cameo_keptn += rep["kept"]
    cameo_rows.append({"split": split, "language": iso, "n_kept": rep["kept"], "n_total": rep["total"],
                       "native_emotions": ", ".join(native), "dropped": ", ".join(dropped) or "—"})
rows.append({
    "dataset": "CAMEO", "study": 2, "language": ", ".join(sorted(cameo_langs)),
    "modalities": MODALITIES["CAMEO"],
    "native_emotions": ", ".join(sorted(cameo_native)),
    "canonical_emotions": ", ".join(sorted(cameo_kept)),
    "dropped": ", ".join(sorted(cameo_dropped)) or "—",
    "n_kept": cameo_keptn, "n_total": cameo_total, "notes": NOTES["CAMEO"] + f" ({len(cameo_langs)} languages, 13 splits).",
})

out = s.results_root / "tables"
out.mkdir(parents=True, exist_ok=True)
cols = ["dataset", "study", "language", "modalities", "native_emotions",
        "canonical_emotions", "dropped", "n_kept", "n_total", "notes"]
with (out / "datasets_overview.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
with (out / "cameo_splits.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["split", "language", "n_kept", "n_total", "native_emotions", "dropped"])
    w.writeheader(); w.writerows(cameo_rows)

# markdown (printed + saved)
def md_table(rows_, cols_):
    lines = ["| " + " | ".join(cols_) + " |", "|" + "|".join(["---"] * len(cols_)) + "|"]
    for r in rows_:
        lines.append("| " + " | ".join(str(r[c]) for c in cols_) + " |")
    return "\n".join(lines)

md = "# Datasets overview\n\n" + md_table(rows, cols) + "\n\n## CAMEO splits (Study 2)\n\n" + \
     md_table(cameo_rows, ["split", "language", "n_kept", "n_total", "native_emotions", "dropped"]) + \
     "\n\n\\* MELD face deferred (multi-speaker frames).\n"
(out / "datasets_overview.md").write_text(md)
print(md)
print(f"\n-> {out}/datasets_overview.{{md,csv}} + cameo_splits.csv")
