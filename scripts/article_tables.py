"""Curated publication-ready tables from the Study-1 analysis bundle.

Emits results/tables/article/{cross_modal_kappa,modality_baselines}.{tex,md} — clean,
selected rows (clean/headline models only), formatted for the paper. Run AFTER analyze.

Number formatting follows app.report.numfmt:
accuracy/F1 are percent-scale with the unit in the column header, kappa stays a 0-1
fraction, missing values render as "---".

  PYTHONPATH=. .venv/bin/python scripts/article_tables.py
"""
from pathlib import Path
import pandas as pd

from app.analysis.results import read_json
from app.config.settings import Settings
from app.report import numfmt
from app.report.tables import latex_table

s = Settings()
b = read_json(s.results_root / "analysis" / "study1" / "bundle.json")
OUT = s.results_root / "tables" / "article"
OUT.mkdir(parents=True, exist_ok=True)

# clean/headline model per modality per corpus
TEXT = {"nEMO": "mling_tabularis", "RAVDESS": "en_text_hartmann", "MELD": "en_text_hartmann"}
SPEECH = "speech_zeroshot"
FACE = "en_face_vit"
CORPORA = ["nEMO", "RAVDESS", "MELD"]


def write(df, stem, headers, caption, column_spec=None):
    (OUT / f"{stem}.md").write_text(df.to_markdown(index=False))
    (OUT / f"{stem}.tex").write_text(
        latex_table(df, headers, caption=caption, column_spec=column_spec))


def _ci(ci) -> str:
    """CI bracket at the same precision as the kappa value it bounds (fmt_kappa, 3dp)."""
    return f"[{numfmt.fmt_kappa(ci.lo)}, {numfmt.fmt_kappa(ci.hi)}]" if ci else numfmt.NA


# ---- Table 1: cross-modal kappa (clean models) ----
def _agr(corpus, a, b_):
    pair = {a, b_}
    for x in b.agreements:
        if x.corpus == corpus and {x.model_a, x.model_b} == pair:
            return x.kappa, _ci(x.ci), x.n_used
    return None


rows = []
def _fleiss(corpus):
    for f in b.fleiss:
        if f.corpus == corpus:
            return f.kappa, _ci(f.ci), f.n_used
    return None


for c in CORPORA:
    for a, bb, lab in [(TEXT[c], SPEECH, "text–speech"), (TEXT[c], FACE, "text–face"),
                       (SPEECH, FACE, "speech–face")]:
        r = _agr(c, a, bb)
        if r is None:
            continue
        k, ci, n = r
        rows.append({"corpus": c, "coef": "Cohen", "modalities": lab,
                     "κ": numfmt.fmt_kappa(k), "95% CI": ci, "n": n})
    # trimodal Fleiss (text–speech–face), where the clean trio exists (RAVDESS, MELD)
    fr = _fleiss(c)
    if fr is not None:
        fk, fci, fn = fr
        rows.append({"corpus": c, "coef": "Fleiss", "modalities": "text–speech–face",
                     "κ": numfmt.fmt_kappa(fk), "95% CI": fci, "n": fn})
k_df = pd.DataFrame(rows)
k_headers = [numfmt.unit_header("Corpus"), numfmt.unit_header("Coefficient"),
             numfmt.unit_header("Modalities"), numfmt.unit_header("κ"),
             r"\textbf{95\% CI}", numfmt.unit_header("n")]
write(k_df, "cross_modal_kappa", k_headers,
      "Cross-modal agreement (Cohen's κ for pairs, Fleiss' κ for the trimodal triad) "
      "on clean/headline models, Study 1.", column_spec="lllllr")

# ---- Table 2: per-modality baselines (clean models) ----
bl = {(x.corpus, x.model_id): x for x in b.baselines}
rows2 = []
for c in CORPORA:
    row = {"corpus": c}
    for modlab, mid in [("text", TEXT[c]), ("speech", SPEECH), ("face", FACE)]:
        x = bl.get((c, mid))
        row[f"{modlab} acc"] = numfmt.to_percent(x.accuracy, 2) if x else numfmt.NA
        row[f"{modlab} F1"] = numfmt.to_percent(x.macro_f1, 2) if x else numfmt.NA
    rows2.append(row)
bdf = pd.DataFrame(rows2)
b_headers = [numfmt.unit_header("Corpus"),
             numfmt.unit_header("Text accuracy", numfmt.PERCENT_UNIT),
             numfmt.unit_header("Text F1", numfmt.PERCENT_UNIT),
             numfmt.unit_header("Speech accuracy", numfmt.PERCENT_UNIT),
             numfmt.unit_header("Speech F1", numfmt.PERCENT_UNIT),
             numfmt.unit_header("Face accuracy", numfmt.PERCENT_UNIT),
             numfmt.unit_header("Face F1", numfmt.PERCENT_UNIT)]
write(bdf, "modality_baselines", b_headers,
      "Per-modality accuracy / macro-F1 (clean/headline models), Study 1.",
      column_spec="lllllll")

print("=== cross_modal_kappa ===\n" + k_df.to_markdown(index=False))
print("\n=== modality_baselines ===\n" + bdf.to_markdown(index=False))
print(f"\n-> {OUT}/*.{{tex,md}}")
