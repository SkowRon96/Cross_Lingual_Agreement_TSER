"""Diverging bar: which modality is closer to the true emotion when text and speech disagree.

For the clean/headline text<->speech pair per corpus, split resolved conflicts (exactly one
model right) into speech-closer (left) vs text-closer (right). The flip from speech-dominated
(RAVDESS, acted) to text-dominated (MELD, natural) is the headline of the conflict analysis.

Computes the clean text<->speech conflict counts directly from the cached predictions
(same conflict() logic as the analysis) -> writes results/figures/article/conflict_direction.png.

  PYTHONPATH=. .venv/bin/python scripts/conflict_direction_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

from app.analysis.conflicts import conflict
from app.analysis.run import _load, gold_lookup
from app.config.settings import Settings
from app.predict.store import PredictionStore
from app.report import style

style.apply()
s = Settings()
OUT = s.results_root / "figures" / "article"
OUT.mkdir(parents=True, exist_ok=True)
store = PredictionStore(s.results_root / "predictions")
index_dir = Path(s.index_dir)
# text/speech are non-emotion series -> shades of one hue, NOT categorical slots
# 0/1 (those are byte-identical to the sadness/happiness emotion colours, so a
# reader who learned blue=sadness elsewhere would misread this figure).
SPEECH_C = style.series_shade(0, 2)   # speech (left), dark
TEXT_C = style.series_shade(1, 2)     # text (right), light
EDGE, INK = style.BACKGROUND, style.INK

# clean/headline text model per corpus, speech = zeroshot; labels in English
CLEAN = {
    "RAVDESS": ("en_text_hartmann", "RAVDESS\n(EN, acted)"),
    "nEMO": ("mling_tabularis", "nEMO\n(PL, acted)"),
    "MELD": ("en_text_hartmann", "MELD\n(EN, natural)"),
}
SPEECH = "speech_zeroshot"
ORDER = ["RAVDESS", "nEMO", "MELD"]  # acted -> natural, eye travels down toward text

fig, ax = plt.subplots(figsize=(8.2, 3.2))
for y, corpus in enumerate(ORDER):
    text_m, _ = CLEAN[corpus]
    gold = gold_lookup(1, corpus, index_dir)
    c = conflict(_load(store, 1, corpus, text_m), _load(store, 1, corpus, SPEECH),
                 gold, text_m, SPEECH, corpus)
    text_c, speech_c = c.a_closer_to_gold, c.b_closer_to_gold
    resolved = text_c + speech_c
    text_sh, speech_sh = text_c / resolved, speech_c / resolved
    # speech to the left (negative), text to the right (positive)
    ax.barh(y, -speech_sh, color=SPEECH_C, edgecolor=EDGE, linewidth=1.0)
    ax.barh(y, text_sh, color=TEXT_C, edgecolor=EDGE, linewidth=1.0)
    ax.text(-speech_sh - 0.02, y, f"speech  {speech_c}\n({speech_sh:.0%})",
            ha="right", va="center", fontsize=9, color=INK)
    ax.text(text_sh + 0.02, y, f"{text_c}  text\n({text_sh:.0%})",
            ha="left", va="center", fontsize=9, color=INK)

ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([CLEAN[c][1] for c in ORDER], color=INK)
ax.set_xlim(-1.35, 1.35)
ax.set_xticks([])
# Arrows via mathtext: the dissertation serif (Charter) has no arrow glyphs and would
# render them as tofu boxes; mathtext routes them through STIX, which does.
ax.set_xlabel(r"$\leftarrow$ speech closer to truth            "
              r"text closer to truth $\rightarrow$", color=INK)
ax.grid(False)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "conflict_direction.png", bbox_inches="tight")  # dpi from style.apply() (300)
print(f"-> {OUT}/conflict_direction.png")
