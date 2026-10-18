"""Pipeline architecture diagram in the dissertation's visual system.

Replaces the mermaid-rendered pastel version: same structure, but the
dissertation serif, palette and line weights, so it sits next to the result
figures without reading as a different document.

Three node kinds, three shades of one hue (research_style.series_shade):
  light  -- inputs and outputs (corpora, models, figures, tables)
  mid    -- processing stages (the four pipeline modules)
  dark   -- persistent stores (indexed corpora, prediction cache, results)

  .venv/bin/python scripts/pipeline_architecture_figure.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from app.config.settings import Settings
from app.report import style

style.apply()
OUT = Settings().results_root / "figures" / "article"
OUT.mkdir(parents=True, exist_ok=True)

IO_C, PROC_C, STORE_C = (style.series_shade(i, 3) for i in (2, 1, 0))  # light, mid, dark
INK, MUTED = style.INK, style.MUTED
BW, BH = 17.0, 8.0          # default box size
CYL_W, CYL_H = 17.0, 11.0   # cylinder body


def box(ax, x, y, text, fill, ink=INK, w=BW, h=BH):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.1",
        facecolor=fill, edgecolor=MUTED, linewidth=0.6, zorder=2))
    ax.text(x, y, text, ha="center", va="center", color=ink,
            fontsize=8.5, fontweight="bold", zorder=3, linespacing=1.35)
    return x, y, w, h


def cylinder(ax, x, y, text, w=CYL_W, h=CYL_H):
    """A store: rectangle body capped with ellipses, drawn in the dark shade."""
    ry = h * 0.16
    ax.add_patch(Rectangle((x - w / 2, y - h / 2 + ry), w, h - 2 * ry,
                           facecolor=STORE_C, edgecolor="none", zorder=2))
    for cy in (y - h / 2 + ry, y + h / 2 - ry):
        ax.add_patch(Ellipse((x, cy), w, 2 * ry, facecolor=STORE_C,
                             edgecolor="none", zorder=2))
    ax.add_patch(Ellipse((x, y + h / 2 - ry), w, 2 * ry, facecolor="none",
                         edgecolor=style.BACKGROUND, linewidth=0.8, zorder=3))
    ax.text(x, y, text, ha="center", va="center", color=style.BACKGROUND,
            fontsize=8.5, fontweight="bold", zorder=4, linespacing=1.35)
    return x, y, w, h


def arrow(ax, p0, p1, label=None, rad=0.0, label_pos=0.5):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11,
        connectionstyle=f"arc3,rad={rad}", color=INK,
        linewidth=0.9, shrinkA=0, shrinkB=0, zorder=1))
    if label:
        mx = p0[0] + (p1[0] - p0[0]) * label_pos
        my = p0[1] + (p1[1] - p0[1]) * label_pos
        ax.text(mx, my + 2.2, label, ha="center", va="bottom", color=MUTED,
                fontsize=6.8, zorder=4,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=style.BACKGROUND,
                          edgecolor="none"))


fig, ax = plt.subplots(figsize=(12.5, 5.2))
ax.set_xlim(0, 134); ax.set_ylim(0, 62); ax.axis("off"); ax.grid(False)

# columns
c1, c2, c3, c4, c5 = 11, 38, 70, 98, 123

# --- nodes -------------------------------------------------------------
box(ax, c1, 50, "DATASETS", IO_C)
box(ax, c1, 31, "CLASSIFICATION\nMODELS", IO_C)
box(ax, c1, 11, "EXTERNAL MODELS\n/ PREDICTIONS", IO_C)

box(ax, c2, 50, "INGESTOR", PROC_C)
box(ax, c2, 31, "PREDICTOR", PROC_C)

cylinder(ax, c3, 50, "INDEXED\nDATASETS")
cylinder(ax, c3, 15, "PREDICTIONS")

box(ax, c4, 38, "ANALYZER", PROC_C)
box(ax, c4, 13, "REPORT\nGENERATOR", PROC_C)

cylinder(ax, c5, 38, "ANALYSIS\nRESULTS")
box(ax, c5, 17, "FIGURES", IO_C, w=13, h=6.5)
box(ax, c5, 6, "TABLES", IO_C, w=13, h=6.5)

# --- edges -------------------------------------------------------------
arrow(ax, (c1 + BW / 2, 50), (c2 - BW / 2, 50), "load")
arrow(ax, (c2 + BW / 2, 50), (c3 - CYL_W / 2, 50), "harmonisation")
arrow(ax, (c1 + BW / 2, 31), (c2 - BW / 2, 31))
# indexed corpora feed the predictor (sample ids + gold)
arrow(ax, (c3 - CYL_W / 2, 46), (c2 + BW / 2, 34), rad=0.18)
arrow(ax, (c2 + BW / 2, 28), (c3 - CYL_W / 2, 18), "cache predictions", rad=-0.12)
arrow(ax, (c1 + BW / 2, 11), (c3 - CYL_W / 2, 13), "import predictions")
arrow(ax, (c3 + CYL_W / 2, 47), (c4 - BW / 2, 41), rad=-0.12)
arrow(ax, (c3 + CYL_W / 2, 17), (c4 - BW / 2, 35), rad=0.12)
arrow(ax, (c4 + BW / 2, 38), (c5 - CYL_W / 2, 38), "export")
arrow(ax, (c5 - CYL_W / 2, 33), (c4 + BW / 2, 15), rad=0.16)
arrow(ax, (c4 + BW / 2, 14), (c5 - 6.5, 17), rad=-0.10)
arrow(ax, (c4 + BW / 2, 11), (c5 - 6.5, 7), rad=0.10)

fig.savefig(OUT / "pipeline_architecture.png", bbox_inches="tight")
plt.close(fig)
print(f"-> {OUT}/pipeline_architecture.png")
