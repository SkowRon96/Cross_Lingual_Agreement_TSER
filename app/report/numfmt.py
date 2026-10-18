"""Number formatting for the dissertation results chapter.

Single source of truth for how numbers are rendered in LaTeX tables and
prose; this module is the binding spec -- read it before changing anything
here. The convention was reverse-engineered from the dissertation's earlier
results chapters, not invented, so changes should be checked against the
typeset chapters rather than against taste.

Rules implemented:

- Decimal point is always ``.``, never locale-dependent (``f"{x:.2f}"``, not
  ``locale``-aware formatting).
- Classification metrics (accuracy, precision, recall, macro F1) get 2
  decimal places; coarser-grained quantities (annotation-audit rates,
  latency, throughput, sizes) get 1.
- Missing values render as ``"---"`` (which LaTeX ligates into an em dash),
  never an empty cell and never ``"n/a"``.
- Ranges use ``"--"`` (LaTeX en dash), never a hyphen (a hyphen is a
  word-break point and is indistinguishable from a minus sign at
  ``\\scriptsize``).
- ``\\phantom{0}`` left-pads shorter integer parts to align a column under
  ``C{}`` centering -- but ONLY when the column actually contains values of
  differing integer-part width; at uniform width it must NOT be added, since
  it would shift every value off the column's true center.
- A column's shared unit goes in the header (``[\\%]``), never in the cell;
  a unit that is NOT shared across the whole column stays at the value
  instead (see ``unit_header``).
- Identifiers get LaTeX-escaped: ``_`` -> ``\\_``, ``/`` -> ``\\slash``.

Kappa (Cohen's / Fleiss') is reported as a 0-1 fraction, NOT percent-scaled
like the other metrics, because it is an agreement coefficient rather than a
rate. ``fmt_kappa`` nonetheless keeps the precision as a parameter instead of
hardcoding it: it defaults to 3 decimals (the conservative choice, since
reducing precision silently discards information), and callers that need a
different width pass ``decimals`` explicitly.
"""
from __future__ import annotations

NA = "---"  # LaTeX source for "not applicable" -- ligates to an em dash.

_RANGE_SEP = "--"  # LaTeX en dash (two ASCII hyphens), never a plain hyphen.

_PHANTOM = r"\phantom{0}"


def _is_na(value) -> bool:
    if value is None:
        return True
    try:
        return value != value  # NaN != NaN, true for float('nan') and numpy nan
    except TypeError:
        return False


def fmt_number(value, decimals: int = 2) -> str:
    """Format a number with a fixed decimal count and a literal dot.

    Returns ``NA`` ("---") for ``None`` or NaN instead of raising or
    printing "nan".
    """
    if _is_na(value):
        return NA
    return f"{float(value):.{decimals}f}"


def fmt_metric(value, decimals: int = 2) -> str:
    """Classification metrics: accuracy, precision, recall, macro F1.

    Two decimal places by default, per format_liczb_w_wynikach.md.
    """
    return fmt_number(value, decimals)


def fmt_coarse(value, decimals: int = 1) -> str:
    """Coarser-grained quantities: annotation-audit rates, latency,
    throughput, sizes. One decimal place by default.
    """
    return fmt_number(value, decimals)


def fmt_kappa(value, decimals: int = 3) -> str:
    """Format Cohen's / Fleiss' kappa.

    UNRESOLVED by the author: whether kappa should be shown as a 0-1
    fraction or converted to the percent scale is an open decision (see
    module docstring). This defaults to a 0-1 fraction at 3 decimals;
    pass ``decimals`` explicitly (or call ``to_percent`` yourself first) once
    that decision is made, rather than hardcoding a new default here.
    """
    return fmt_number(value, decimals)


def fmt_range(lo, hi, decimals: int = 2) -> str:
    """Format a numeric range with a LaTeX en dash: ``"15.6--71.1"``.

    Returns ``NA`` if either bound is missing.
    """
    if _is_na(lo) or _is_na(hi):
        return NA
    return f"{fmt_number(lo, decimals)}{_RANGE_SEP}{fmt_number(hi, decimals)}"


def to_percent(value, decimals: int = 2) -> str:
    """Convert a 0-1 fraction metric to a bare percent-scale cell string.

    ``0.9601 -> "96.01"``. The ``%``/``[\\%]`` sign is NOT included here --
    it belongs in the column header (see ``unit_header``) or, in prose, is
    appended by the caller as ``\\%``.
    """
    if _is_na(value):
        return NA
    return fmt_number(float(value) * 100, decimals)


def unit_header(label: str, unit: str | None = None) -> str:
    """Build a bold LaTeX table header, appending ``[unit]`` only when that
    unit is shared by the whole column.

    Pass ``unit=None`` (default) when the column mixes units (e.g. a
    literature-review "Result" column mixing F1/accuracy across datasets) --
    then the unit stays at the value instead (format each value with its own
    unit suffix rather than calling this with a unit).
    """
    if unit is None:
        return rf"\textbf{{{label}}}"
    return rf"\textbf{{{label} [{unit}]}}"


PERCENT_UNIT = r"\%"


def escape_underscore(text: str) -> str:
    """Escape underscores for LaTeX (``arch_bilstm`` -> ``arch\\_bilstm``)."""
    return text.replace("_", r"\_")


def escape_slash(text: str) -> str:
    """Escape slashes for LaTeX (``a/b`` -> ``a\\slash b``).

    A bare ``/`` is not a linebreak opportunity in LaTeX, so a value like
    ``disgust/neutral/sadness`` becomes one unbreakable word that can
    overflow a narrow column. ``\\slash`` is a proper (breakable) LaTeX
    command; note the trailing space, needed so the following letter doesn't
    get absorbed into the command name.
    """
    return text.replace("/", r"\slash ")


def escape_identifier(text: str) -> str:
    """Escape a raw identifier (run name, model id, class list, ...) for a
    LaTeX table cell: underscores and slashes both handled.
    """
    return escape_slash(escape_underscore(text))


def _int_part_width(formatted: str) -> int:
    """Width (in characters) of the leading number's integer part.

    Counts an optional leading ``-`` plus the run of digits before the first
    non-digit (``.`` or a range separator). Used to compare widths across a
    column, not to render anything by itself.
    """
    i = 0
    if i < len(formatted) and formatted[i] == "-":
        i += 1
    while i < len(formatted) and formatted[i].isdigit():
        i += 1
    return i


def phantom_pad_column(values: list[str]) -> list[str]:
    """Left-pad values with ``\\phantom{0}`` to align integer-part widths.

    Per format_liczb_w_wynikach.md, ``\\phantom{0}`` is inserted only when
    the column, taken as a whole, actually mixes integer-part widths (e.g.
    "6.7--53.3" alongside "15.6--71.1"). At uniform width nothing is added --
    padding a uniform column would push every value off the column's true
    center. This is checked over the whole column, never a single value in
    isolation. ``NA`` ("---") entries are left untouched and excluded from
    the width comparison.
    """
    widths = [_int_part_width(v) for v in values if v != NA]
    if not widths or max(widths) == min(widths):
        return list(values)

    target = max(widths)
    out = []
    for v in values:
        if v == NA:
            out.append(v)
            continue
        pad = _PHANTOM * (target - _int_part_width(v))
        out.append(pad + v)
    return out
