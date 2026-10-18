"""Color system for the dissertation results chapter.

Single source of truth for every figure/table color in the results chapter.
This module is the binding spec — read it before changing anything here. The
hex values are already flattened on a white background, so they paste cleanly
into LaTeX/PowerPoint/Figma even where alpha channels don't survive.

This module is intentionally separate from
``app.config.emotions.EMOTION_COLORS`` (an Okabe-Ito colorblind-safe palette
used by the older figure generators). The two palettes are NOT interchangeable
and not every generator has been migrated. Do not mix hexes from the two
palettes on the same figure.

EMOTION CLASS ORDER IS FIXED. ``EMOTION_ORDER`` reuses
``app.config.emotions.CANONICAL_ORDER`` (anger, disgust, fear, happiness,
sadness, surprise, neutral) rather than the grouped, human-readable order the
palette is usually tabulated in (neutral first). CANONICAL_ORDER is already the project's single source of truth for
class order everywhere else (confusion matrices, per-class tables, legends);
reusing it here keeps this module's figures positionally aligned with every
other figure in the dissertation. If you ever change this order, every figure
in every chapter that assumes a fixed class position (side-by-side confusion
matrices, stacked bars, per-class tables) desynchronizes. Don't.
"""
from __future__ import annotations

from matplotlib.colors import Colormap, LinearSegmentedColormap

from app.config.emotions import CANONICAL_ORDER, Emotion

# ---------------------------------------------------------------------------
# Emotion palette: 7 classes x 3 alpha levels.
#
# Levels are "base" (alpha 1.00, opaque), "0.71" and "0.42" -- the three
# alpha values plot_grouped_bars(per_category_color=True) uses for a
# three-series group (0.42 + 0.58 * i / (n-1), n=3). Hexes below are those
# alphas already flattened onto white (#FFFFFF).
# ---------------------------------------------------------------------------

EMOTION_ORDER: list[Emotion] = list(CANONICAL_ORDER)  # fixed -- see module docstring

ALPHA_LEVELS: tuple[str, ...] = ("base", "0.71", "0.42")

EMOTION_PALETTE: dict[Emotion, dict[str, str]] = {
    Emotion.NEUTRAL: {"base": "#7D8A99", "0.71": "#A3ACB7", "0.42": "#C8CED4"},
    Emotion.HAPPINESS: {"base": "#F0A92B", "0.71": "#F4C268", "0.42": "#F9DBA6"},
    Emotion.SADNESS: {"base": "#2B7FB8", "0.71": "#68A4CD", "0.42": "#A6C9E1"},
    Emotion.FEAR: {"base": "#7A4FA3", "0.71": "#A182BE", "0.42": "#C7B5D8"},
    Emotion.ANGER: {"base": "#D1453B", "0.71": "#DE7B74", "0.42": "#ECB1AD"},
    Emotion.DISGUST: {"base": "#2F9E63", "0.71": "#6BBA90", "0.42": "#A8D6BD"},
    Emotion.SURPRISE: {"base": "#E0559B", "0.71": "#E986B8", "0.42": "#F2B8D5"},
}


def emotion_color(emotion: Emotion | str, level: str = "base") -> str:
    """Return the flattened hex for an emotion class at a given alpha level.

    ``emotion`` may be an ``Emotion`` member or its string value (e.g.
    ``"anger"``). ``level`` is one of ``"base"`` (alpha 1.00), ``"0.71"`` or
    ``"0.42"``.

    Fails loudly (``KeyError``) on an unknown emotion name or level instead
    of silently returning a grey/neutral fallback -- a silent fallback would
    hide a typo'd emotion label inside a figure instead of crashing the
    build where it can be caught.
    """
    if isinstance(emotion, Emotion):
        key = emotion
    else:
        try:
            key = Emotion(emotion)
        except ValueError as exc:
            raise KeyError(
                f"Unknown emotion class {emotion!r}; expected one of "
                f"{[e.value for e in EMOTION_ORDER]}"
            ) from exc

    try:
        levels = EMOTION_PALETTE[key]
    except KeyError as exc:
        raise KeyError(f"No palette entry for emotion {key!r}") from exc

    try:
        return levels[level]
    except KeyError as exc:
        raise KeyError(
            f"Unknown alpha level {level!r} for {key!r}; expected one of {ALPHA_LEVELS}"
        ) from exc


# ---------------------------------------------------------------------------
# Categorical palette: 8 slots x 3 alpha levels, for series that are not
# emotions. Slots 0-6 intentionally reuse the emotion hexes (see
# paleta_kolorow.md); slot 7 is an extra brown for an 8th series. If a figure
# mixes emotion series and other series, give the other series shades of one
# color instead of pulling the next categorical index (per the source doc).
# ---------------------------------------------------------------------------

CATEGORICAL_PALETTE: list[dict[str, str]] = [
    {"base": "#2B7FB8", "0.71": "#68A4CD", "0.42": "#A6C9E1"},  # 0
    {"base": "#F0A92B", "0.71": "#F4C268", "0.42": "#F9DBA6"},  # 1
    {"base": "#D1453B", "0.71": "#DE7B74", "0.42": "#ECB1AD"},  # 2
    {"base": "#2F9E63", "0.71": "#6BBA90", "0.42": "#A8D6BD"},  # 3
    {"base": "#7A4FA3", "0.71": "#A182BE", "0.42": "#C7B5D8"},  # 4
    {"base": "#E0559B", "0.71": "#E986B8", "0.42": "#F2B8D5"},  # 5
    {"base": "#7D8A99", "0.71": "#A3ACB7", "0.42": "#C8CED4"},  # 6
    {"base": "#8C6239", "0.71": "#AD9072", "0.42": "#CFBDAC"},  # 7
]


def categorical_color(index: int, level: str = "base") -> str:
    """Return the flattened hex for categorical series ``index`` (0-7).

    Fails loudly on an out-of-range index or unknown alpha level.
    """
    if not isinstance(index, int) or not (0 <= index < len(CATEGORICAL_PALETTE)):
        raise IndexError(
            f"Categorical palette has {len(CATEGORICAL_PALETTE)} entries "
            f"(indices 0-{len(CATEGORICAL_PALETTE) - 1}); got {index!r}"
        )
    entry = CATEGORICAL_PALETTE[index]
    try:
        return entry[level]
    except KeyError as exc:
        raise KeyError(
            f"Unknown alpha level {level!r} for index {index}; expected one of {ALPHA_LEVELS}"
        ) from exc


# ---------------------------------------------------------------------------
# Dataset-split shades: one sequential ramp (light -> dark) for
# training/validation/test, used when series are variants of the same thing
# (splits, seeds, ablation arms) and should read as shades, not categories.
# ---------------------------------------------------------------------------

SPLIT_COLORS: dict[str, str] = {
    "training": "#B2D1E3",
    "validation": "#6398BA",
    "test": "#264D71",
}


def split_color(split: str) -> str:
    """Return the shade for a dataset split ("training", "validation", "test").

    Fails loudly on any other name -- no fuzzy aliasing (e.g. "train" is not
    silently accepted) so a typo doesn't quietly fall through to a KeyError
    somewhere less obvious.
    """
    try:
        return SPLIT_COLORS[split]
    except KeyError as exc:
        raise KeyError(
            f"Unknown split {split!r}; expected one of {tuple(SPLIT_COLORS)}"
        ) from exc


# Shades of one hue, light -> dark, for series that are NOT emotions.
# Positions 0-6 of CATEGORICAL_PALETTE are byte-identical to the emotion
# colours, so using them for modalities or corpora would make one colour mean
# "sadness" in a confusion matrix and "speech" in the next figure. The palette
# document resolves that clash by giving non-emotion series shades of a single
# colour instead of successive categorical positions.
SERIES_SHADES: list[str] = ["#264D71", "#6398BA", "#B2D1E3"]


def series_shade(index: int, n_series: int = 2) -> str:
    """Return a shade for series `index` of `n_series` non-emotion series.

    Two series get the extremes of the ramp (maximum separation); three get all
    three stops. Raises for more than three -- at that point shades of one hue
    stop being distinguishable and the figure needs rethinking, which should be
    a loud decision rather than a silently reused colour.
    """
    if not 0 <= index < n_series:
        raise IndexError(f"index {index} out of range for {n_series} series")
    if n_series == 2:
        return [SERIES_SHADES[0], SERIES_SHADES[2]][index]
    if n_series == 3:
        return SERIES_SHADES[index]
    raise ValueError(
        f"{n_series} series do not separate as shades of one hue; "
        "use a different encoding (facets, line style) instead"
    )


# ---------------------------------------------------------------------------
# Helper colors.
# ---------------------------------------------------------------------------

INK = "#14202B"        # text, axes
MUTED = "#7D8A99"       # secondary labels/captions (matches neutral's base)
GRID = "#C8D0D8"        # gridlines
BACKGROUND = "#FFFFFF"  # figure/table background

# ---------------------------------------------------------------------------
# Ramps. Node points are interpolated into matplotlib LinearSegmentedColormaps.
#
# SEQUENTIAL: non-negative quantities (counts, attention weights, coverage).
# DIVERGING: signed quantities only (differences, residuals, model weights) --
# never non-negative data, since the ramp's midpoint encodes zero and would
# suggest a split in one-sided data that isn't there.
# ---------------------------------------------------------------------------

SEQUENTIAL_NODES: list[str] = ["#FBFCFD", "#CFE3EE", "#7FB3D0", "#3A6F9A", "#152F4D"]
DIVERGING_NODES: list[str] = ["#1F4D7D", "#7FA8C4", "#F7F3EA", "#E0A08E", "#8C2020"]


def _cmap(name: str, nodes: list[str]) -> Colormap:
    return LinearSegmentedColormap.from_list(name, nodes)


SEQUENTIAL: Colormap = _cmap("research_sequential", SEQUENTIAL_NODES)
DIVERGING: Colormap = _cmap("research_diverging", DIVERGING_NODES)
