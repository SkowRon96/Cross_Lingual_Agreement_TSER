import pytest

from app.report import numfmt as nf


def test_fmt_metric_two_decimals_default():
    assert nf.fmt_metric(0.9601) == "0.96"
    assert nf.fmt_metric(1) == "1.00"


def test_fmt_coarse_one_decimal_default():
    assert nf.fmt_coarse(6.66) == "6.7"
    assert nf.fmt_coarse(15.0) == "15.0"


def test_dot_is_always_decimal_separator():
    assert "." in nf.fmt_metric(3.14159)
    assert "," not in nf.fmt_metric(3.14159)


def test_fmt_number_returns_em_dash_for_missing_values():
    assert nf.fmt_number(None) == "---"
    assert nf.fmt_number(float("nan")) == "---"
    assert nf.fmt_metric(None) == "---"
    assert nf.fmt_coarse(None) == "---"


def test_fmt_range_uses_en_dash_not_hyphen():
    assert nf.fmt_range(15.6, 71.1, decimals=1) == "15.6--71.1"
    assert nf.fmt_range(6.66, 53.3, decimals=1) == "6.7--53.3"


def test_fmt_range_missing_bound_is_na():
    assert nf.fmt_range(None, 5.0) == "---"
    assert nf.fmt_range(5.0, float("nan")) == "---"


def test_to_percent_conversion():
    assert nf.to_percent(0.9601) == "96.01"
    assert nf.to_percent(0.5) == "50.00"
    assert nf.to_percent(1.0) == "100.00"


def test_to_percent_missing_is_na():
    assert nf.to_percent(None) == "---"


def test_fmt_kappa_default_is_fraction_three_decimals():
    assert nf.fmt_kappa(0.4231) == "0.423"


def test_fmt_kappa_decimals_is_a_parameter_not_hardcoded():
    # The presentation of kappa is an open author decision -- verify the
    # function actually honors an overridden precision rather than a fixed
    # default baked into the formatting.
    assert nf.fmt_kappa(0.4231, decimals=2) == "0.42"
    assert nf.fmt_kappa(0.4231, decimals=0) == "0"


def test_unit_header_appends_unit_when_shared():
    assert nf.unit_header("Test set", nf.PERCENT_UNIT) == r"\textbf{Test set [\%]}"
    assert nf.unit_header("F1", nf.PERCENT_UNIT) == r"\textbf{F1 [\%]}"


def test_unit_header_omits_unit_when_none():
    assert nf.unit_header("Result") == r"\textbf{Result}"


def test_escape_underscore():
    assert nf.escape_underscore("arch-bilstm_self_attention") == r"arch-bilstm\_self\_attention"


def test_escape_slash():
    assert nf.escape_slash("disgust/neutral/sadness") == r"disgust\slash neutral\slash sadness"


def test_escape_identifier_handles_both():
    assert nf.escape_identifier("a_b/c_d") == r"a\_b\slash c\_d"


def test_phantom_pad_column_enabled_on_mixed_widths():
    col = ["6.7--53.3", "15.6--71.1"]
    padded = nf.phantom_pad_column(col)
    assert padded == [r"\phantom{0}6.7--53.3", "15.6--71.1"]


def test_phantom_pad_column_enabled_on_mixed_single_values():
    col = ["1.1", "38.9"]
    padded = nf.phantom_pad_column(col)
    assert padded == [r"\phantom{0}1.1", "38.9"]


def test_phantom_pad_column_disabled_on_uniform_widths():
    col = ["16.7", "15.6", "71.1"]  # all integer parts are 2 digits
    padded = nf.phantom_pad_column(col)
    assert padded == col  # unchanged, no phantom inserted


def test_phantom_pad_column_ignores_na_entries():
    col = ["1.1", "---", "38.9"]
    padded = nf.phantom_pad_column(col)
    assert padded == [r"\phantom{0}1.1", "---", "38.9"]


def test_phantom_pad_column_na_only_or_empty_is_noop():
    assert nf.phantom_pad_column(["---", "---"]) == ["---", "---"]
    assert nf.phantom_pad_column([]) == []


def test_na_constant_is_em_dash_not_blank_or_na_string():
    assert nf.NA == "---"
    assert nf.NA != ""
    assert nf.NA.lower() != "n/a"
