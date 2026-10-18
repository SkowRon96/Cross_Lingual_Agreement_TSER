"""Study-3 article figure: risk-coverage curve for the S2 (modality-priority) arbitration
strategy, one series per corpus (headline variant: clean channels, speech = Qwen zero-shot).

Reads results/analysis/study3/bundle.json (ArbitrationBundle.selective[strategy="S2"].curve,
already computed by app.analysis.arbitration -- accept order: agreeing samples first, then
descending fused confidence, D-H). Nothing is recomputed from predictions here.

Axis units follow the project's unit rule: coverage is a composition/cutoff quantity
-> percent axis; risk (= 1 - selective accuracy) is a metric -> 0-1 fraction axis.
Corpora are a non-emotion categorical series -> categorical palette. Colors/INK/GRID
come exclusively from app.report.style (backed by app.report.research_style). No title
on the figure -- per project convention titles go in the
chapter text, not on the image; the proposed caption is printed to stdout instead.

Coverage points 100/90/80/70% (the reporting grid fixed for this study)
are marked as ticks plus a point marker on every curve, read from
SelectiveResult.selective_accuracy (risk = 1 - accuracy at that coverage), not recomputed.

Emits results/figures/article/risk_coverage.png.

  PYTHONPATH=. .venv/bin/python scripts/risk_coverage_figure.py
"""
import matplotlib.pyplot as plt

from app.analysis.arbitration import COVERAGE_POINTS, read_json
from app.config.settings import Settings
from app.report import style

style.apply()
s = Settings()
BUNDLE_PATH = s.results_root / "analysis" / "study3" / "bundle.json"
OUT = s.results_root / "figures" / "article"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGY = "S2"  # the proposed arbitration strategy (modality priority), not S0/S1/S3
CORPORA = ["nEMO", "RAVDESS", "MELD"]
# 3 non-emotion series -> shades of one hue (categorical slots double as emotion colours)
# as article_figures.py / conflict_direction_figure.py.
CORPUS_COLOR = {c: style.series_shade(i, 3) for i, c in enumerate(CORPORA)}

bundle = read_json(BUNDLE_PATH)


def _selective(corpus: str):
    block = next(b for b in bundle.corpora if b.variant == "headline" and b.corpus == corpus)
    return next(sel for sel in block.selective if sel.strategy == STRATEGY)


fig, ax = plt.subplots(figsize=(7, 5))

for corpus in CORPORA:
    sel = _selective(corpus)
    color = CORPUS_COLOR[corpus]
    xs = [p.coverage * 100 for p in sel.curve]   # coverage -> percent axis
    ys = [p.risk for p in sel.curve]              # risk -> 0-1 fraction axis
    ax.plot(xs, ys, color=color, lw=1.8, label=corpus)
    # mark the required coverage points, reading accuracy straight from the bundle
    for c in COVERAGE_POINTS:
        acc = sel.selective_accuracy[f"{c:.2f}"]
        ax.plot(c * 100, 1.0 - acc, marker="o", markersize=5.5,
                color=color, markeredgecolor=style.INK, markeredgewidth=0.8, zorder=5)

ax.set_xlabel("Share of samples the system answers [%]")
ax.set_ylabel("Error rate among answered samples")
ax.set_xlim(0, 100)
# Headroom above the highest curve (nEMO plateaus ~0.75) is reserved for the legend, so the
# legend box sits in empty space instead of covering any series (checked visually below).
ax.set_ylim(0, 0.95)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
for c in COVERAGE_POINTS:
    ax.axvline(c * 100, color=style.GRID, lw=0.8, linestyle=":", zorder=0)
ax.grid(True, axis="y", color=style.GRID, alpha=0.55)
ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.99), fontsize=9, framealpha=0.95)

style.save_fig(fig, OUT / "risk_coverage")
print(f"-> {OUT}/risk_coverage.png")
for corpus in CORPORA:
    sel = _selective(corpus)
    print(f"{corpus}: AURC={sel.aurc:.4f} acc_full={sel.accuracy_full_coverage:.4f} "
          f"acc_agree={sel.accuracy_agree_subset:.4f} acc_conflict={sel.accuracy_conflict_subset:.4f} "
          f"selective_accuracy={sel.selective_accuracy}")
