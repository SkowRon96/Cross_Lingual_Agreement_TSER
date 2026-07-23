"""report orchestration: analysis JSON -> tables/ + figures/ (deterministic, overwrites)."""
from pathlib import Path

from app.analysis.results import read_json
from app.report import figures, style, tables


def report_study(study, *, results_root: Path) -> list[Path]:
    results_root = Path(results_root)
    bundle = read_json(results_root / "analysis" / f"study{study}" / "bundle.json")
    style.apply()
    tdir = results_root / "tables" / f"study{study}"
    fdir = results_root / "figures" / f"study{study}"
    written: list[Path] = []
    for name, df in [("kappa", tables.kappa_table(bundle)),
                     ("baselines", tables.baseline_table(bundle)),
                     ("pl_vs_en", tables.pl_vs_en_table(bundle)),
                     ("study2", tables.study2_table(bundle))]:
        if df.empty:
            continue
        tables.write_table(df, tdir / name)
        written += [tdir / f"{name}.{e}" for e in ("csv", "md", "tex", "png")]
    for b in bundle.baselines:
        if b.confusion:
            style.save_fig(figures.confusion_fig(b), fdir / f"cm_{b.corpus}_{b.model_id}")
            written += [fdir / f"cm_{b.corpus}_{b.model_id}.{e}" for e in ("pdf", "png")]
    if bundle.agreements:
        style.save_fig(figures.kappa_ci_fig(bundle), fdir / "kappa_ci")
        written += [fdir / f"kappa_ci.{e}" for e in ("pdf", "png")]
    for c in bundle.conflicts:
        style.save_fig(figures.conflict_fig(c), fdir / f"conflict_{c.corpus}")
        written += [fdir / f"conflict_{c.corpus}.{e}" for e in ("pdf", "png")]
    if bundle.study2:
        style.save_fig(figures.study2_scatter(bundle), fdir / "study2_scatter")
        written += [fdir / f"study2_scatter.{e}" for e in ("pdf", "png")]
    return written
