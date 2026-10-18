import pytest
from matplotlib.colors import Colormap

from app.config.emotions import CANONICAL_ORDER, Emotion
from app.report import research_style as rs


def test_emotion_order_matches_canonical_order():
    # The palette's fixed class order must track the project-wide single
    # source of truth, not the row order printed in paleta_kolorow.md.
    assert rs.EMOTION_ORDER == list(CANONICAL_ORDER)


def test_emotion_palette_has_all_seven_classes_and_three_levels():
    assert set(rs.EMOTION_PALETTE) == set(CANONICAL_ORDER)
    for levels in rs.EMOTION_PALETTE.values():
        assert set(levels) == {"base", "0.71", "0.42"}


def test_emotion_colors_match_spec_hexes():
    assert rs.emotion_color(Emotion.ANGER) == "#D1453B"
    assert rs.emotion_color("anger", "0.71") == "#DE7B74"
    assert rs.emotion_color("anger", "0.42") == "#ECB1AD"
    assert rs.emotion_color(Emotion.NEUTRAL) == "#7D8A99"
    assert rs.emotion_color("happiness") == "#F0A92B"
    assert rs.emotion_color("sadness") == "#2B7FB8"
    assert rs.emotion_color("fear") == "#7A4FA3"
    assert rs.emotion_color("disgust") == "#2F9E63"
    assert rs.emotion_color("surprise") == "#E0559B"


def test_emotion_color_accepts_enum_and_string_equally():
    assert rs.emotion_color(Emotion.FEAR) == rs.emotion_color("fear")


def test_emotion_color_fails_loud_on_unknown_name():
    with pytest.raises(KeyError):
        rs.emotion_color("joy")


def test_emotion_color_fails_loud_on_unknown_level():
    with pytest.raises(KeyError):
        rs.emotion_color("anger", "0.99")


def test_categorical_palette_eight_entries_matching_emotions_for_first_seven():
    assert len(rs.CATEGORICAL_PALETTE) == 8
    assert rs.categorical_color(0) == "#2B7FB8"  # sadness
    assert rs.categorical_color(7) == "#8C6239"  # extra brown, not an emotion


def test_categorical_color_fails_loud_on_out_of_range_index():
    with pytest.raises(IndexError):
        rs.categorical_color(8)
    with pytest.raises(IndexError):
        rs.categorical_color(-1)


def test_categorical_color_fails_loud_on_unknown_level():
    with pytest.raises(KeyError):
        rs.categorical_color(0, "nope")


def test_split_colors_exact_hexes():
    assert rs.split_color("training") == "#B2D1E3"
    assert rs.split_color("validation") == "#6398BA"
    assert rs.split_color("test") == "#264D71"


def test_split_color_fails_loud_on_unknown_split():
    with pytest.raises(KeyError):
        rs.split_color("train")  # no fuzzy aliasing


def test_helper_colors_exact_hexes():
    assert rs.INK == "#14202B"
    assert rs.MUTED == "#7D8A99"
    assert rs.GRID == "#C8D0D8"
    assert rs.BACKGROUND == "#FFFFFF"


def test_sequential_and_diverging_are_colormaps():
    assert isinstance(rs.SEQUENTIAL, Colormap)
    assert isinstance(rs.DIVERGING, Colormap)
    # Endpoints should reproduce the node colors (as RGBA floats).
    import matplotlib.colors as mcolors

    assert rs.SEQUENTIAL(0.0) == pytest.approx(mcolors.to_rgba("#FBFCFD"))
    assert rs.SEQUENTIAL(1.0) == pytest.approx(mcolors.to_rgba("#152F4D"))
    assert rs.DIVERGING(0.0) == pytest.approx(mcolors.to_rgba("#1F4D7D"))
    assert rs.DIVERGING(1.0) == pytest.approx(mcolors.to_rgba("#8C2020"))


def test_sequential_and_diverging_are_distinct_ramps():
    assert rs.SEQUENTIAL_NODES != rs.DIVERGING_NODES
