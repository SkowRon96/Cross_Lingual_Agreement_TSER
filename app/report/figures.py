"""Figure builders: confusion matrix, kappa CI, conflicts, Study-2 scatter.

Each returns a matplotlib Figure (never calls plt.show). Fixed class order + palette from
app.report.style (research_style-backed) so every artefact is visually consistent.
"""
import matplotlib.pyplot as plt
import numpy as np

from app.analysis.results import AnalysisBundle, ConflictResult, Baseline
from app.report import style

style.apply()

_ORDER = [e.value for e in style.ORDER]

# Non-emotion two-group contrasts (headline vs secondary, model A vs model B, ...)
# get shades from the categorical palette rather than emotion colors.
_HEADLINE_C = style.categorical_color(0)   # blue
_SECONDARY_C = style.MUTED                 # grey


def confusion_fig(baseline: Baseline):
    cols = _ORDER + (["abstain"] if any(
        "abstain" in row for row in baseline.confusion.values()) else [])
    mat = np.zeros((len(_ORDER), len(cols)), dtype=int)
    for gi, g in enumerate(_ORDER):
        row = baseline.confusion.get(g, {})
        for cj, c in enumerate(cols):
            mat[gi, cj] = row.get(c, 0)
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(cols), 1.2 + 0.8 * len(_ORDER)))
    # Confusion counts are non-negative -> sequential ramp (never diverging).
    im = ax.imshow(mat, cmap=style.SEQUENTIAL)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(_ORDER)), _ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(f"{baseline.model_id} on {baseline.corpus} (n={baseline.n})")
    thresh = mat.max() / 2 if mat.max() else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color=style.BACKGROUND if mat[i, j] > thresh else style.INK, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def kappa_ci_fig(bundle: AnalysisBundle):
    # headline pairs first, then the rest; each as a point with a CI whisker
    rows = sorted(bundle.agreements, key=lambda a: (not a.headline, a.corpus, a.model_a))
    labels = [f"{a.corpus}: {a.model_a}↔{a.model_b}" for a in rows]
    ks = [a.kappa for a in rows]
    los = [a.ci.lo if a.ci else a.kappa for a in rows]
    his = [a.ci.hi if a.ci else a.kappa for a in rows]
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.4 * max(len(rows), 1)))
    y = np.arange(len(rows))
    err = np.array([[k - lo for k, lo in zip(ks, los)],
                    [hi - k for k, hi in zip(ks, his)]])
    colors = [_HEADLINE_C if a.headline else _SECONDARY_C for a in rows]
    ax.errorbar(ks, y, xerr=err if len(rows) else None, fmt="o", ecolor=style.GRID,
                capsize=3, linestyle="none")
    for yi, (k, a) in enumerate(zip(ks, rows)):
        ax.plot(k, yi, "o", color=colors[yi])
    ax.set_yticks(y, labels)
    ax.axvline(0.0, color=style.INK, lw=0.5)
    ax.set_xlabel("Cohen's κ (95% bootstrap CI)")
    ax.set_xlim(-0.2, 1.0)
    ax.set_title(f"Cross-modal agreement (study {bundle.study})")
    return fig


def conflict_fig(conflict: ConflictResult):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    cats = ["a_closer", "b_closer", "both_wrong"]
    vals = [conflict.a_closer_to_gold, conflict.b_closer_to_gold, conflict.both_wrong]
    # a vs b are two non-emotion series (categorical), "both_wrong" is neither -> muted grey.
    cat_colors = [style.categorical_color(0), style.categorical_color(2), style.MUTED]
    ax1.bar(cats, vals, color=cat_colors)
    ax1.set_title(f"{conflict.corpus}: who is closer to gold\n"
                  f"({conflict.model_a} vs {conflict.model_b}, n={conflict.n_conflicts})")
    ax1.set_ylabel("conflicts")
    emos = [e for e in _ORDER if e in conflict.per_emotion]
    counts = [conflict.per_emotion[e] for e in emos]
    bar_colors = [style.emotion_color(e) for e in emos]
    ax2.bar(emos, counts, color=bar_colors)
    ax2.set_title("conflicts per gold emotion")
    ax2.set_xticks(range(len(emos)), emos, rotation=45, ha="right")
    return fig


def study2_scatter(bundle: AnalysisBundle):
    rows = [r for r in bundle.study2 if r.syntactic_distance is not None]
    fig, ax = plt.subplots(figsize=(7, 5))
    # PL vs rest is a two-group categorical contrast, not an emotion or a split.
    pl_color, rest_color = style.categorical_color(2), style.categorical_color(0)
    for r in rows:
        is_pl = r.language == "pl"
        ax.scatter(r.syntactic_distance, r.kappa,
                   s=40 + 30 * (r.resource_tier or 0),
                   color=pl_color if is_pl else rest_color,
                   edgecolor=style.INK, zorder=3 if is_pl else 2)
        ax.annotate(r.language, (r.syntactic_distance, r.kappa),
                    textcoords="offset points", xytext=(4, 4), fontsize=8,
                    fontweight="bold" if is_pl else "normal", color=style.INK)
    ax.set_xlabel("syntactic distance from Polish (URIEL/lang2vec)")
    ax.set_ylabel("text↔speech Cohen's κ")
    ax.set_title("Study 2: agreement vs typological distance (PL highlighted)\n"
                 "marker size = resource tier")
    return fig
