"""Curated publication-ready tables from the Study-1 analysis bundle.

Emits results/tables/article/{cross_modal_kappa,modality_baselines}.{tex,md,csv} — clean,
selected rows (clean/headline models only), formatted for the paper. Run AFTER analyze.

  PYTHONPATH=. .venv/bin/python scripts/article_tables.py
"""
from pathlib import Path
import pandas as pd

from app.analysis.results import read_json
from app.config.settings import Settings

s = Settings()
b = read_json(s.results_root / "analysis" / "study1" / "bundle.json")
OUT = s.results_root / "tables" / "article"
OUT.mkdir(parents=True, exist_ok=True)

# clean/headline model per modality per corpus
TEXT = {"nEMO": "mling_tabularis", "RAVDESS": "en_text_hartmann", "MELD": "en_text_hartmann"}
SPEECH = "speech_zeroshot"
FACE = "en_face_vit"
CORPORA = ["nEMO", "RAVDESS", "MELD"]


def write(df, stem, caption):
    (OUT / f"{stem}.md").write_text(df.to_markdown(index=False))
    (OUT / f"{stem}.tex").write_text(df.to_latex(index=False, caption=caption, escape=True))


# ---- Table 1: cross-modal kappa (clean models) ----
def _agr(corpus, a, b_):
    pair = {a, b_}
    for x in b.agreements:
        if x.corpus == corpus and {x.model_a, x.model_b} == pair:
            ci = f"[{x.ci.lo:.2f}, {x.ci.hi:.2f}]" if x.ci else "—"
            return x.kappa, ci, x.n_used
    return None


rows = []
def _fleiss(corpus):
    for f in b.fleiss:
        if f.corpus == corpus:
            ci = f"[{f.ci.lo:.2f}, {f.ci.hi:.2f}]" if f.ci else "—"
            return f.kappa, ci, f.n_used
    return None


for c in CORPORA:
    for a, bb, lab in [(TEXT[c], SPEECH, "text–speech"), (TEXT[c], FACE, "text–face"),
                       (SPEECH, FACE, "speech–face")]:
        r = _agr(c, a, bb)
        if r is None:
            continue
        k, ci, n = r
        rows.append({"corpus": c, "coef": "Cohen", "modalities": lab,
                     "κ": f"{k:.3f}", "95% CI": ci, "n": n})
    # trimodal Fleiss (text–speech–face), where the clean trio exists (RAVDESS, MELD)
    fr = _fleiss(c)
    if fr is not None:
        fk, fci, fn = fr
        rows.append({"corpus": c, "coef": "Fleiss", "modalities": "text–speech–face",
                     "κ": f"{fk:.3f}", "95% CI": fci, "n": fn})
k_df = pd.DataFrame(rows)
write(k_df, "cross_modal_kappa",
      "Cross-modal agreement (Cohen's κ for pairs, Fleiss' κ for the trimodal triad) "
      "on clean/headline models, Study 1.")

# ---- Table 2: per-modality baselines (clean models) ----
bl = {(x.corpus, x.model_id): x for x in b.baselines}
rows2 = []
for c in CORPORA:
    row = {"corpus": c}
    for modlab, mid in [("text", TEXT[c]), ("speech", SPEECH), ("face", FACE)]:
        x = bl.get((c, mid))
        row[f"{modlab} acc"] = f"{x.accuracy:.2f}" if x else "—"
        row[f"{modlab} F1"] = f"{x.macro_f1:.2f}" if x else "—"
    rows2.append(row)
bdf = pd.DataFrame(rows2)
write(bdf, "modality_baselines",
      "Per-modality accuracy / macro-F1 (clean/headline models), Study 1.")

print("=== cross_modal_kappa ===\n" + k_df.to_markdown(index=False))
print("\n=== modality_baselines ===\n" + bdf.to_markdown(index=False))
print(f"\n-> {OUT}/*.{{tex,md}}")
