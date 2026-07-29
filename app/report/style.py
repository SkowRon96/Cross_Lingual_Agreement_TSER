"""Shared report style: palette, class order, figure/table save helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config.emotions import CANONICAL_ORDER, EMOTION_COLORS

PALETTE = EMOTION_COLORS
ORDER = CANONICAL_ORDER

# Muted, low-saturation palette matching the architecture diagram (classDef fills).
# Pale fills paired with a single soft edge/label colour for a uniform, subdued look.
MUTED = {
    "teal": "#DDEEE7",
    "green": "#E6F2E3",
    "tan": "#F4E8D5",
    "lavender": "#E7E2F3",
}
MUTED_EDGE = "#7a7a7a"    # soft grey outline for pastel bars
MUTED_INK = "#3a3a3a"     # dark grey for value labels / axis text


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "figure.autolayout": True,
    })


def save_fig(fig, path_stem) -> None:
    from pathlib import Path
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def table_to_png(df, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(min(2 + 1.2 * len(df.columns), 20),
                                    0.5 + 0.35 * (len(df) + 1)))
    ax.axis("off")
    t = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns),
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
