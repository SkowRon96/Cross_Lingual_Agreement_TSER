"""Table builders + writers (Markdown, LaTeX, PNG).

Numeric formatting and LaTeX escaping route through ``app.report.numfmt``
(the single source of truth for the number conventions):
classification metrics (accuracy, macro F1) are percent-scale with the unit
in the column header; kappa stays a 0-1 fraction; missing values render as
``numfmt.NA`` ("---"); identifiers get LaTeX-escaped.

``latex_table`` renders a booktabs table from cells/headers that are already
formatted+escaped strings -- it does no formatting itself. This sidesteps
pandas' own ``to_latex(escape=True)``, which cannot coexist with literal
LaTeX markup (``\\textbf{...}``) in column headers (pandas/Styler treats a
custom header as a format string and chokes on the braces).
"""
from pathlib import Path

import pandas as pd

from app.analysis.results import AnalysisBundle
from app.report import numfmt, style


def latex_table(df: pd.DataFrame, headers: list[str], caption: str | None = None,
                 column_spec: str | None = None) -> str:
    """Render a booktabs LaTeX table.

    ``df`` cells must already be final display strings (formatted via
    ``numfmt`` and, where they may contain identifiers, escaped via
    ``numfmt.escape_identifier``). ``headers`` are literal LaTeX header text
    (e.g. built with ``numfmt.unit_header``), one per column -- supplied
    separately from ``df.columns`` so header markup never round-trips
    through pandas' own escaping.
    """
    if len(headers) != len(df.columns):
        raise ValueError(
            f"headers has {len(headers)} entries, df has {len(df.columns)} columns")
    spec = column_spec or ("l" * len(df.columns))
    lines = ["\\begin{table}"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# Column-name -> (display label, unit) used to build LaTeX headers generically
# for the per-study tables below. Columns not listed fall back to a
# Title Case of their name with no unit.
_LABELS: dict[str, str] = {
    "kappa": "κ", "ci_lo": "CI low", "ci_hi": "CI high",
    "n_used": "n (used)", "n_excl": "n (excluded)",
    "model_a": "Model A", "model_b": "Model B",
    "macro_f1": "Macro F1", "syn_dist": "Syntactic distance",
    "tier": "Resource tier", "p_value": "p", "statistic": "Statistic",
}
_UNITS: dict[str, str] = {
    "accuracy": numfmt.PERCENT_UNIT,
    "macro_f1": numfmt.PERCENT_UNIT,
}


def _header_for(col: str) -> str:
    label = _LABELS.get(col, col.replace("_", " ").title())
    return numfmt.unit_header(label, _UNITS.get(col))


def write_table(df: pd.DataFrame, stem) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(df.to_markdown(index=False))

    tex_df = df.astype(str)
    for col in tex_df.columns:
        tex_df[col] = tex_df[col].map(numfmt.escape_identifier)
    headers = [_header_for(c) for c in df.columns]
    stem.with_suffix(".tex").write_text(latex_table(tex_df, headers))

    style.table_to_png(df, stem.with_suffix(".png"))


def kappa_table(bundle: AnalysisBundle) -> pd.DataFrame:
    rows = [{"corpus": a.corpus, "model_a": a.model_a, "model_b": a.model_b,
             "kappa": numfmt.fmt_kappa(a.kappa),
             "ci_lo": numfmt.fmt_kappa(a.ci.lo) if a.ci else numfmt.NA,
             "ci_hi": numfmt.fmt_kappa(a.ci.hi) if a.ci else numfmt.NA,
             "n_used": a.n_used, "n_excl": a.n_excluded,
             "headline": a.headline, "secondary": a.secondary_flag}
            for a in bundle.agreements]
    return pd.DataFrame(rows)


def baseline_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"model": b.model_id, "corpus": b.corpus, "modality": b.modality,
                          "n": b.n, "accuracy": numfmt.to_percent(b.accuracy, 2),
                          "macro_f1": numfmt.to_percent(b.macro_f1, 2)} for b in bundle.baselines])


def study2_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"split": r.split, "language": r.language,
                          "kappa": numfmt.fmt_kappa(r.kappa), "n_used": r.n_used,
                          "syn_dist": numfmt.fmt_number(r.syntactic_distance, 2),
                          "tier": r.resource_tier}
                         for r in bundle.study2])


def pl_vs_en_table(bundle: AnalysisBundle) -> pd.DataFrame:
    return pd.DataFrame([{"test": t.name, "corpus": t.corpus, "detail": t.detail,
                          "statistic": numfmt.fmt_number(t.statistic, 3),
                          "p_value": numfmt.fmt_number(t.p_value, 4)}
                         for t in bundle.tests])
