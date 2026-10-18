"""Study-3 article table: arbitration strategies (S0-S3) x corpora, headline variant.

Emits results/tables/article/arbitration.{md,tex} -- one row per (corpus, strategy),
plus a post-audit nEMO section reported as mean +- sd over the three seeds
(audit_s7/s17/s27, decision D-I of app.analysis.arbitration). Every number is read back
from results/analysis/study3/bundle.json through the ArbitrationBundle pydantic schema;
nothing here is computed from raw predictions and nothing is invented.

S2_insample is the leaky in-sample diagnostic (app.analysis.arbitration docstring: "not a
result") and is deliberately excluded -- only S0/S1/S2/S3 are shown.

Number formatting follows app.report.numfmt:
accuracy/F1 are percent-scale with the unit in the column header, missing values render
as "---". Delta-accuracy-vs-S0 is also percent-scale (percentage points) with an explicit
sign, via a small local helper -- numfmt itself has no signed formatter and is not modified
here.

  PYTHONPATH=. .venv/bin/python scripts/arbitration_table.py
"""
import numpy as np
import pandas as pd

from app.analysis.arbitration import AUDIT_SEEDS, read_json
from app.config.settings import Settings
from app.report import numfmt
from app.report.tables import latex_table

s = Settings()
BUNDLE_PATH = s.results_root / "analysis" / "study3" / "bundle.json"
OUT = s.results_root / "tables" / "article"
OUT.mkdir(parents=True, exist_ok=True)

CORPORA = ["nEMO", "RAVDESS", "MELD"]
STRATEGIES = ["S0", "S1", "S2", "S3"]  # S2_insample excluded: leaky diagnostic, not a result
# Descriptive names carry the table; the S-codes stay internal to the bundle schema.
STRATEGY_NAMES = {"S0": "Single-modality", "S1": "Confidence-based",
                  "S2": "Conflict-aware", "S3": "Oracle"}
AUDIT_VARIANTS = [f"audit_s{seed}" for seed in AUDIT_SEEDS]

bundle = read_json(BUNDLE_PATH)


def _block(variant: str, corpus: str):
    for b in bundle.corpora:
        if b.variant == variant and b.corpus == corpus:
            return b
    raise KeyError(f"no block for variant={variant!r} corpus={corpus!r}")


def _fmt_delta_pp(value: float | None) -> str:
    """Signed percentage points, 2dp: 0.1417 -> '+14.17', -0.0439 -> '-4.39', 0.0 -> '0.00'."""
    if value is None:
        return numfmt.NA
    pp = round(value * 100, 2)
    if pp == 0.0:
        return numfmt.fmt_number(0.0, 2)
    sign = "+" if pp > 0 else ""
    return f"{sign}{numfmt.fmt_number(pp, 2)}"


# ---- one merged table: channel weights + strategy results, one row per corpus ----
# Weights and strategies live in the same table because the weights explain the
# strategies: an asterisk marks a channel whose weight exceeds the sum of all the
# others, which is exactly the condition under which a degenerate one-hot vector wins
# every sample and S2 collapses onto that channel.
MODALITIES = ["text", "speech", "face"]


def _mean_weights(blocks) -> dict[str, float]:
    """Mean fitted weight per modality, averaged over folds (and over seeds if several)."""
    out = {}
    for m in MODALITIES:
        vals = [fw.weights[m] for b in blocks for fw in b.fold_weights if m in fw.weights]
        if vals:
            out[m] = float(np.mean(vals))
    return out


def _weight_cells(w: dict[str, float]) -> dict[str, str]:
    """Percent-scale weights; '*' on the channel outweighing all the others combined."""
    dominant = next((m for m, v in w.items()
                     if v > sum(x for k, x in w.items() if k != m)), None)
    return {m: (numfmt.to_percent(w[m], 2) + ("*" if m == dominant else ""))
            if m in w else numfmt.NA
            for m in MODALITIES}


def _strategy_cells(blocks) -> dict[str, str]:
    """'accuracy / macro-F1' per strategy; mean +- sd when several seed blocks are given."""
    cells = {}
    for strat in STRATEGIES:
        accs = [next(r.accuracy for r in b.strategies if r.strategy == strat) for b in blocks]
        f1s = [next(r.macro_f1 for r in b.strategies if r.strategy == strat) for b in blocks]
        if len(blocks) == 1:
            cells[strat] = f"{numfmt.to_percent(accs[0], 2)} / {numfmt.to_percent(f1s[0], 2)}"
        else:
            cells[strat] = (f"{np.mean(accs) * 100:.2f} $\\pm$ {np.std(accs, ddof=1) * 100:.2f}"
                            f" / {np.mean(f1s) * 100:.2f} $\\pm$ {np.std(f1s, ddof=1) * 100:.2f}")
    return cells


rows = []
for corpus in CORPORA:
    blocks = [_block("headline", corpus)]
    row = {"corpus": corpus}
    row.update(_weight_cells(_mean_weights(blocks)))
    row.update(_strategy_cells(blocks))
    row["n"] = blocks[0].n_used
    rows.append(row)

# post-audit nEMO: three seeds, weights and strategies averaged over them (decision D-I)
audit_blocks = [_block(v, "nEMO") for v in AUDIT_VARIANTS]
row = {"corpus": "nEMO (post-audit, 3 seeds)"}
row.update(_weight_cells(_mean_weights(audit_blocks)))
row.update(_strategy_cells(audit_blocks))
n_vals = {b.n_used for b in audit_blocks}
row["n"] = n_vals.pop() if len(n_vals) == 1 else numfmt.NA
rows.append(row)

df = pd.DataFrame(rows)[["corpus", *MODALITIES, *STRATEGIES, "n"]].rename(
    columns=STRATEGY_NAMES)
(OUT / "arbitration.md").write_text(df.to_markdown(index=False))

headers = [
    numfmt.unit_header("Dataset"),
    numfmt.unit_header("Text", numfmt.PERCENT_UNIT),
    numfmt.unit_header("Speech", numfmt.PERCENT_UNIT),
    numfmt.unit_header("Face", numfmt.PERCENT_UNIT),
    *(numfmt.unit_header(STRATEGY_NAMES[st], numfmt.PERCENT_UNIT) for st in STRATEGIES),
    numfmt.unit_header("n"),
]
caption = (
    "Modality arbitration, headline variant (clean channels; speech = Qwen2-Audio "
    "zero-shot). The first three columns give the channel weights fitted for S2 -- each "
    "channel's accuracy on the conflict subset, normalised to sum to 100, averaged over "
    "five out-of-fold fits. An asterisk marks a channel whose weight exceeds the sum of "
    "all the others: there a degenerate one-hot speech vector wins every sample, which is "
    "why the conflict-aware strategy reduces exactly to the single-modality one on nEMO "
    "and RAVDESS, but not on MELD. The strategy "
    "columns give accuracy / macro-F1. Single-modality: the best individual channel. "
    "Confidence-based: argmax of the summed harmonised score vectors. Conflict-aware: the "
    "same sum weighted by the fitted channel weights. Oracle: correct whenever any "
    "modality is correct; not an implementable method, it marks the ceiling of late fusion. The post-audit row repeats the analysis with the post-audit RoBERTa "
    "text model (speech unchanged), as mean $\\pm$ sample sd over 3 training seeds."
)
(OUT / "arbitration.tex").write_text(
    latex_table(df, headers, caption=caption, column_spec="lrrrllllr"))

print(df.to_markdown(index=False))
print(f"\n-> {OUT}/arbitration.{{md,tex}}")
