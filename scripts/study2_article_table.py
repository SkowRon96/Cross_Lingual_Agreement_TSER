"""Study-2 article table: per-language accuracy/macro-F1 (text + speech) and text<->speech κ,
with MELD added as a natural-English reference point.

Text model = mling_tabularis (multilingual, same across all languages), speech = speech_zeroshot
(zero-shot). CAMEO splits come from the Study-2 bundle; MELD is pulled from the Study-1 bundle
using the SAME multilingual text model so the row is comparable (κ = 0.183, not the English-only
en_text_hartmann's 0.202). Emits results/tables/article/study2_langs.{md,tex}.

Number formatting follows app.report.numfmt:
accuracy/F1 are percent-scale with the unit in the column header, kappa stays a 0-1
fraction, missing values render as "---", corpus identifiers with underscores (e.g.
crema_d, jl_corpus) are LaTeX-escaped.

  PYTHONPATH=. .venv/bin/python scripts/study2_article_table.py
"""
import pandas as pd

from app.analysis.results import read_json
from app.config.settings import Settings
from app.report import numfmt
from app.report.tables import latex_table

s = Settings()
OUT = s.results_root / "tables" / "article"
OUT.mkdir(parents=True, exist_ok=True)
TEXT, SPEECH = "mling_tabularis", "speech_zeroshot"

b2 = read_json(s.results_root / "analysis" / "study2" / "bundle.json")
b1 = read_json(s.results_root / "analysis" / "study1" / "bundle.json")


def _fmt_ci(ci):
    return f"[{numfmt.fmt_kappa(ci.lo)}, {numfmt.fmt_kappa(ci.hi)}]" if ci else numfmt.NA


def _row(corpus, language, bl_by, k, ci, n, note=""):
    t, sp = bl_by.get((corpus, TEXT)), bl_by.get((corpus, SPEECH))
    return {
        "language": language, "corpus": corpus + (f" {note}" if note else ""),
        "text acc": numfmt.to_percent(t.accuracy, 2) if t else numfmt.NA,
        "text F1": numfmt.to_percent(t.macro_f1, 2) if t else numfmt.NA,
        "speech acc": numfmt.to_percent(sp.accuracy, 2) if sp else numfmt.NA,
        "speech F1": numfmt.to_percent(sp.macro_f1, 2) if sp else numfmt.NA,
        "κ text–speech": numfmt.fmt_kappa(k), "95% CI": _fmt_ci(ci), "n": n,
    }


bl2 = {(x.corpus, x.model_id): x for x in b2.baselines}
rows = [_row(r.split, r.language, bl2, r.kappa, r.ci, r.n_used) for r in b2.study2]

# MELD as natural-EN reference (Study-1, same multilingual text model)
bl1 = {(x.corpus, x.model_id): x for x in b1.baselines}
meld_k = next(a for a in b1.agreements
              if a.corpus == "MELD" and {a.model_a, a.model_b} == {TEXT, SPEECH})
rows.append(_row("MELD", "en", bl1, meld_k.kappa, meld_k.ci, meld_k.n_used, note="(natural)"))

df = pd.DataFrame(rows)
# Corpus identifiers can contain underscores (crema_d, jl_corpus, ...) -> escape for LaTeX.
tex_df = df.copy()
tex_df["corpus"] = tex_df["corpus"].map(numfmt.escape_identifier)

(OUT / "study2_langs.md").write_text(df.to_markdown(index=False))
headers = [
    numfmt.unit_header("Language"), numfmt.unit_header("Corpus"),
    numfmt.unit_header("Text accuracy", numfmt.PERCENT_UNIT),
    numfmt.unit_header("Text F1", numfmt.PERCENT_UNIT),
    numfmt.unit_header("Speech accuracy", numfmt.PERCENT_UNIT),
    numfmt.unit_header("Speech F1", numfmt.PERCENT_UNIT),
    numfmt.unit_header("κ text–speech"), r"\textbf{95\% CI}", numfmt.unit_header("n"),
]
(OUT / "study2_langs.tex").write_text(latex_table(
    tex_df, headers, column_spec="llllllllr",
    caption=("Study 2: per-language accuracy / macro-F1 (multilingual text model vs zero-shot "
             "speech) and text↔speech Cohen's κ. MELD added as a natural-English reference "
             "(same multilingual text model as the CAMEO splits).")))
print(df.to_markdown(index=False))
print(f"\n-> {OUT}/study2_langs.{{md,tex}}")
