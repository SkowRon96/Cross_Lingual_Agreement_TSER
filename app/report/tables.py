"""Table builders + writers (Markdown, LaTeX, PNG)."""
from pathlib import Path

import pandas as pd

from app.analysis.results import AnalysisBundle
from app.report import style


def write_table(df: pd.DataFrame, stem) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(df.to_markdown(index=False))
    stem.with_suffix(".tex").write_text(df.to_latex(index=False))
    style.table_to_png(df, stem.with_suffix(".png"))


def kappa_table(bundle: AnalysisBundle) -> pd.DataFrame:
    rows = [{"corpus": a.corpus, "model_a": a.model_a, "model_b": a.model_b,
             "kappa": round(a.kappa, 3),
             "ci_lo": round(a.ci.lo, 3) if a.ci else None,
             "ci_hi": round(a.ci.hi, 3) if a.ci else None,
             "n_used": a.n_used, "n_excl": a.n_excluded,
             "headline": a.headline, "secondary": a.secondary_flag}
            for a in bundle.agreements]
    return pd.DataFrame(rows)


def baseline_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"model": b.model_id, "corpus": b.corpus, "modality": b.modality,
                          "n": b.n, "accuracy": round(b.accuracy, 3),
                          "macro_f1": round(b.macro_f1, 3)} for b in bundle.baselines])


def study2_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"split": r.split, "language": r.language,
                          "kappa": round(r.kappa, 3), "n_used": r.n_used,
                          "syn_dist": r.syntactic_distance, "tier": r.resource_tier}
                         for r in bundle.study2])


def pl_vs_en_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"test": t.name, "corpus": t.corpus, "detail": t.detail,
                          "statistic": round(t.statistic, 3), "p_value": round(t.p_value, 4)}
                         for t in bundle.tests])
