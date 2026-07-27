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
from app.config.emotions import CANONICAL_ORDER, EMOTION_COLORS
from app.config.settings import Settings
from app.predict.store import PredictionStore
from app.report import style
from collections import Counter

style.apply()
s = Settings()
store = PredictionStore(s.results_root / "predictions")
OUT = s.results_root / "figures" / "article"
ORDER = [e.value for e in CANONICAL_ORDER]
BLUE, GREY, ORANGE = "#0072B2", "#999999", "#D55E00"


def _preds(study, corpus, mid):
    return store.read(study, corpus, mid)


# ---------- D1: headline kappa bar ----------
def d1():
    cfg = [("nEMO", 1, "mling_tabularis", "speech_zeroshot", "nEMO\n(PL, acted)", GREY),
           ("RAVDESS", 1, "en_text_hartmann", "speech_zeroshot", "RAVDESS\n(EN, acted)", GREY),
           ("MELD", 1, "en_text_hartmann", "speech_zeroshot", "MELD\n(EN, natural)", ORANGE)]
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
    ax.bar(x, ks, color=colors, yerr=[los, his], capsize=5, edgecolor="black", width=0.6)
    for xi, k in zip(x, ks):
        ax.text(xi, k + 0.015, f"{k:.3f}", ha="center", fontweight="bold")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cohen's κ (text ↔ speech)")
    ax.set_title("Cross-modal agreement rises only with emotionally-loaded text")
    ax.set_ylim(-0.03, max(his[i] + ks[i] for i in range(len(ks))) + 0.06)
    style.save_fig(fig, OUT / "d1_headline_kappa")


# ---------- D2: Qwen accuracy spread + negative collapse (honest: NOT a clean EN/non-EN split) ----------
_LANG_COLOR = {"en": "#0072B2", "pl": "#D55E00", "de": "#009E73", "fr": "#CC79A7",
               "it": "#E69F00", "es": "#56B4E9", "ru": "#F0E442", "bn": "#999999"}


def d2():
    corpora = [(1, "RAVDESS", "en"), (1, "MELD", "en"), (1, "nEMO", "pl")]
    for split in studies.STUDY2_SPLITS:
        corpora.append((2, split, studies.SPLIT_LANG[split]))
    data = []  # (acc, label, iso, Counter preds)
    for study, corpus, iso in corpora:
        if not store.path(study, corpus, "speech_zeroshot").exists():
            continue
        gold = gold_lookup(study, corpus, s.index_dir)
        preds = _preds(study, corpus, "speech_zeroshot")
        b = baseline(preds, gold, "speech_zeroshot", corpus, "speech")
        c = Counter(p.pred.value for p in preds if p.pred)
        data.append((b.accuracy, f"{corpus} ({iso})", iso, c))
    data.sort(key=lambda t: t[0], reverse=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    y = np.arange(len(data))
    ax1.barh(y, [d[0] for d in data], color=[_LANG_COLOR[d[2]] for d in data], edgecolor="black")
    ax1.set_yticks(y, [d[1] for d in data], fontsize=8); ax1.invert_yaxis()
    ax1.set_xlabel("Qwen2-Audio zero-shot accuracy")
    ax1.set_title("Highly uneven across corpora (0.24–0.85) — not a clean language split\n"
                  "(German pavoque high; English emns/jl_corpus low)")
    ax1.axvline(1/7, color="k", ls=":", lw=0.8, label="chance (1/7)")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=v, label=k) for k, v in _LANG_COLOR.items()],
               fontsize=7, ncol=2, title="language")
    # per-emotion prediction share: EASY corpora (acc>=0.6) vs HARD (acc<0.4)
    easy, hard = Counter(), Counter()
    for acc, _, _, c in data:
        (easy if acc >= 0.6 else hard if acc < 0.4 else Counter()).update(c)
    def share(c):
        t = sum(c.values()) or 1
        return [c.get(e, 0) / t for e in ORDER]
    w = 0.4; xe = np.arange(len(ORDER))
    ax2.bar(xe - w/2, share(easy), w, label="easy corpora (acc≥0.6)", color="#009E73", edgecolor="black")
    ax2.bar(xe + w/2, share(hard), w, label="hard corpora (acc<0.4)", color="#D55E00", edgecolor="black")
    ax2.set_xticks(xe, ORDER, rotation=45, ha="right")
    ax2.set_ylabel("share of Qwen predictions")
    ax2.set_title("On hard corpora Qwen collapses to fear/sadness/anger")
    ax2.legend(fontsize=8)
    style.save_fig(fig, OUT / "d2_qwen_accuracy_collapse")
    print(f"D2 easy-corpora pred share: {dict(zip(ORDER,[round(x,2) for x in share(easy)]))}")
    print(f"D2 hard-corpora pred share: {dict(zip(ORDER,[round(x,2) for x in share(hard)]))}")


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
            im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
            ax.set_xlabel("predicted"); ax.set_ylabel("gold")
            ax.set_title(f"{title}\nen_text_hartmann, acc={acc:.2f}")
            for i in range(len(ORDER)):
                for j in range(len(cols)):
                    if m[i, j] > 0.01:
                        ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                                color="white" if m[i, j] > 0.5 else "black", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall")
        fig.savefig((OUT / "d3_text_cm_contrast").with_suffix(".pdf"))
        fig.savefig((OUT / "d3_text_cm_contrast").with_suffix(".png"), dpi=300)
        plt.close(fig)


import sys
_only = sys.argv[1] if len(sys.argv) > 1 else None
if _only == "d1": d1()
elif _only == "d2": d2()
elif _only == "d3": d3()
else: d1(); d2(); d3()
print(f"\n-> {OUT}/*.{{pdf,png}}")
