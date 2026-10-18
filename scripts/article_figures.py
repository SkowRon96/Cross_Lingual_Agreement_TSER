"""Article figures (D1-D3) from cached predictions + gold. Writes results/figures/article/*.{pdf,png}.

D1  headline text<->speech kappa: nEMO(PL acted), RAVDESS(EN acted), MELD(EN natural) + bootstrap CI
D2  Qwen cross-lingual: accuracy per corpus (EN vs non-EN) + per-emotion prediction share EN vs rest
D3  acted-vs-natural text: RAVDESS vs MELD text-model confusion (row-normalized) side by side

  PYTHONPATH=. .venv/bin/python scripts/article_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt

from app.analysis.agreement import aligned_pairs, cohen_kappa
from app.analysis.bootstrap import bootstrap_kappa
from app.analysis.metrics import baseline
from app.analysis.run import gold_lookup
from app.config import studies
from app.config.emotions import CANONICAL_ORDER
from app.config.settings import Settings
from app.predict.store import PredictionStore
from app.report import style
from app.report.numfmt import fmt_kappa
from collections import Counter

style.apply()
s = Settings()
store = PredictionStore(s.results_root / "predictions")
OUT = s.results_root / "figures" / "article"
ORDER = [e.value for e in CANONICAL_ORDER]
# Non-emotion two-group contrast -> shades of one hue. Not MUTED/categorical(1):
# those are the neutral and happiness emotion colours respectively.
MUTED_C, HIGHLIGHT_C = style.series_shade(1, 2), style.series_shade(0, 2)


def _preds(study, corpus, mid):
    return store.read(study, corpus, mid)


# ---------- D1: headline kappa bar ----------
def d1():
    cfg = [("nEMO", 1, "mling_tabularis", "speech_zeroshot", "nEMO\n(PL, acted)", MUTED_C),
           ("RAVDESS", 1, "en_text_hartmann", "speech_zeroshot", "RAVDESS\n(EN, acted)", MUTED_C),
           ("MELD", 1, "en_text_hartmann", "speech_zeroshot", "MELD\n(EN, natural)", HIGHLIGHT_C)]
    ks, los, his, labels, colors = [], [], [], [], []
    for corpus, study, a, b, lab, col in cfg:
        gold = gold_lookup(study, corpus, s.index_dir)
        la, lb, _ = aligned_pairs(_preds(study, corpus, a), _preds(study, corpus, b),
                                  gold, studies.SUPPORTED[a], studies.SUPPORTED[b])
        k = cohen_kappa(la, lb)
        ci = bootstrap_kappa(la, lb, n=1000)
        ks.append(k); los.append(k - ci.lo); his.append(ci.hi - k)
        labels.append(lab); colors.append(col)
        print(f"D1 {corpus}: kappa={k:.3f} CI=[{ci.lo:.3f},{ci.hi:.3f}] n={len(la)}")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ks))
    ax.bar(x, ks, color=colors, yerr=[los, his], capsize=5, edgecolor=style.INK, width=0.6)
    for xi, k in zip(x, ks):
        ax.text(xi, k + 0.015, fmt_kappa(k), ha="center", fontweight="bold")
    ax.axhline(0, color=style.INK, lw=0.6)
    ax.set_xticks(x, labels)
    # kappa and the arrow via mathtext (STIX) -- Charter carries neither glyph.
    ax.set_ylabel(r"Cohen's $\kappa$ (text $\leftrightarrow$ speech)")
    ax.set_title("Cross-modal agreement rises only with emotionally-loaded text")
    ax.set_ylim(-0.03, max(his[i] + ks[i] for i in range(len(ks))) + 0.06)
    style.save_fig(fig, OUT / "d1_headline_kappa")


# ---------- D2: Qwen accuracy spread (a) + negative collapse (b) ----------
# 8 languages -> the categorical palette's 8 slots, one language per index (fixed order).
_LANG_ORDER = ["en", "pl", "de", "fr", "it", "es", "ru", "bn"]
# Fill level 0.71, the shared opacity for filled marks across the thesis figures.
_LANG_COLOR = {lang: style.categorical_color(i, "0.71") for i, lang in enumerate(_LANG_ORDER)}
# figure (b) groups are non-emotion series -> shades of one hue (see
# research_style.series_shade): categorical slots 0/1 double as emotion colours.
POP_C, NICHE_C = style.series_shade(0, 2), style.series_shade(1, 2)


def _zeroshot_data():
    """(accuracy, corpus, iso, Counter of predictions) for every CAMEO split, sorted by acc."""
    data = []
    for split in studies.STUDY2_SPLITS:
        if not store.path(2, split, "speech_zeroshot").exists():
            continue
        gold = gold_lookup(2, split, s.index_dir)
        preds = _preds(2, split, "speech_zeroshot")
        b = baseline(preds, gold, "speech_zeroshot", split, "speech")
        c = Counter(p.pred.value for p in preds if p.pred)
        data.append((b.accuracy, split, studies.SPLIT_LANG[split], c))
    data.sort(key=lambda t: t[0], reverse=True)
    return data


def d2a():
    """Accuracy spread across CAMEO corpora, bars coloured by language (no title)."""
    data = _zeroshot_data()
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(data))
    ax.barh(y, [d[0] for d in data], color=[_LANG_COLOR[d[2]] for d in data])
    ax.set_yticks(y, [f"{d[1]} ({d[2]})" for d in data], fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("Qwen2-Audio zero-shot accuracy")
    from matplotlib.patches import Patch
    langs = [d[2] for d in data]
    handles = [Patch(color=_LANG_COLOR[l], label=l) for l in _LANG_COLOR if l in langs]
    ax.legend(handles=handles, fontsize=9, ncol=2, title="language", loc="lower right")
    ax.grid(True, axis="x", color=style.GRID, alpha=0.55)
    style.save_fig(fig, OUT / "d2_qwen_accuracy")


def d2b():
    """Per-emotion prediction share: popular (acc>=0.6) vs niche (acc<0.4) corpora."""
    data = _zeroshot_data()
    pop, niche = Counter(), Counter()
    for acc, _, _, c in data:
        (pop if acc >= 0.6 else niche if acc < 0.4 else Counter()).update(c)
    def share(c):
        t = sum(c.values()) or 1
        return [c.get(e, 0) / t for e in ORDER]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.4; xe = np.arange(len(ORDER))
    ax.bar(xe - w/2, share(pop), w, label="popular datasets (acc≥0.6)", color=POP_C)
    ax.bar(xe + w/2, share(niche), w, label="niche datasets (acc<0.4)", color=NICHE_C)
    ax.set_xticks(xe, ORDER, rotation=45, ha="right")
    ax.set_ylabel("share of Qwen predictions")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", color=style.GRID, alpha=0.55)
    style.save_fig(fig, OUT / "d2_qwen_collapse")
    print(f"D2 popular pred share: {dict(zip(ORDER,[round(x,2) for x in share(pop)]))}")
    print(f"D2 niche pred share: {dict(zip(ORDER,[round(x,2) for x in share(niche)]))}")


def d2():
    d2a(); d2b()


# ---------- D3: acted-vs-natural text confusion ----------
def _cm_rows(study, corpus, mid):
    gold = gold_lookup(study, corpus, s.index_dir)
    b = baseline(_preds(study, corpus, mid), gold, mid, corpus, "text")
    cols = ORDER + (["abstain"] if any("abstain" in r for r in b.confusion.values()) else [])
    m = np.zeros((len(ORDER), len(cols)))
    for gi, g in enumerate(ORDER):
        row = b.confusion.get(g, {}); tot = sum(row.values()) or 1
        for cj, c in enumerate(cols):
            m[gi, cj] = row.get(c, 0) / tot
    return m, cols, b.accuracy


def d3():
    # autolayout (tight_layout) + a shared colorbar fight each other and squash the panels;
    # use constrained_layout with autolayout OFF, and give each panel its own colorbar.
    with plt.rc_context({"figure.autolayout": False}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for ax, (corpus, title) in zip(axes, [("RAVDESS", "RAVDESS (acted, fixed sentences)"),
                                              ("MELD", "MELD (real dialogue)")]):
            m, cols, acc = _cm_rows(1, corpus, "en_text_hartmann")
            # Row-normalized recall is non-negative -> sequential ramp (never diverging).
            im = ax.imshow(m, cmap=style.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
            ax.set_xlabel("predicted"); ax.set_ylabel("gold")
            ax.set_title(f"{title}\nen_text_hartmann, acc={acc:.2f}")
            for i in range(len(ORDER)):
                for j in range(len(cols)):
                    if m[i, j] > 0.01:
                        ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                                color=style.BACKGROUND if m[i, j] > 0.5 else style.INK, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall")
        fig.savefig((OUT / "d3_text_cm_contrast").with_suffix(".png"), dpi=300)
        plt.close(fig)


import sys
_only = sys.argv[1] if len(sys.argv) > 1 else None
if _only == "d1": d1()
elif _only in ("d2", "d2a", "d2b"): {"d2": d2, "d2a": d2a, "d2b": d2b}[_only]()
elif _only == "d3": d3()
else: d1(); d2(); d3()
print(f"\n-> {OUT}/*.png")
