"""Shared report style: palette, class order, figure/table save helpers.

Colors are no longer hardcoded here -- everything is sourced from
``app.report.research_style``, the single source of truth for the
dissertation's visual system. This module just wires research_style into matplotlib's rcParams and
re-exports the lookup helpers so figure scripts can do
``from app.report import style`` and then ``style.emotion_color(...)``,
``style.categorical_color(...)``, etc., without importing research_style
directly everywhere.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config.emotions import CANONICAL_ORDER
from app.report.research_style import (
    BACKGROUND,
    CATEGORICAL_PALETTE,
    DIVERGING,
    EMOTION_ORDER,
    GRID,
    INK,
    MUTED,
    SEQUENTIAL,
    SPLIT_COLORS,
    categorical_color,
    emotion_color,
    split_color,
    series_shade,
)

ORDER = CANONICAL_ORDER

__all__ = [
    "ORDER", "INK", "MUTED", "GRID", "BACKGROUND",
    "SEQUENTIAL", "DIVERGING", "CATEGORICAL_PALETTE", "EMOTION_ORDER",
    "SPLIT_COLORS", "emotion_color", "categorical_color", "split_color", "series_shade",
    "apply", "save_fig", "table_to_png",
]


def apply(base_font_size: float = 10.0) -> None:
    """Install the dissertation's visual system globally. Call before any figure.

    Typography and line weights are fixed here, as part of the committed
    visual system for the whole dissertation -- so figures from this chapter
    sit next to figures from the other chapters without reading as a different
    document. Colours come from app.report.research_style, the single source
    of truth for the palette.

    Charter is the intended serif; matplotlib falls back through the list when
    it is missing. Keep the fallback consistent across the whole thesis rather
    than mixing faces between chapters.
    """
    plt.rcParams.update({
        # typography -- the dissertation's serif stack, STIX for mathtext
        "font.family": "serif",
        "font.serif": ["Charter", "XCharter", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "legend.frameon": False,
        # ground and ink
        "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": MUTED,
        # line weights
        "axes.linewidth": 0.6, "axes.titlepad": 9.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        # grid on the value axis only, per the visual system
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.alpha": 0.55, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        # output -- Type 42 so the thesis PDF embeds TrueType, never Type 3
        "figure.dpi": 300, "savefig.dpi": 300,
        "figure.autolayout": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_fig(fig, path_stem) -> None:
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def table_to_png(df, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(min(2 + 1.2 * len(df.columns), 20),
                                    0.5 + 0.35 * (len(df) + 1)))
    fig.patch.set_facecolor(BACKGROUND)
    ax.axis("off")
    t = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns),
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)
    for cell in t.get_celld().values():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(INK)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
