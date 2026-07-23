"""Guard that the standalone Windows script's prompt+parser stay identical to the app.

run_qwen.py is standalone (imports torch/transformers at module top, so it can't be
imported on the Mac). We extract only its VERBATIM prompt+parser block, exec it in
isolation, and assert functional equivalence with app/models/speech/zeroshot/prompt.py.

Semantics note: the Windows parser returns "" for no-match (written to parquet as an
abstain marker); the app parser returns None. We treat "" and None as equivalent.
"""
from pathlib import Path

from app.models.speech.zeroshot import prompt as canon

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "qwen_windows" / "run_qwen.py"
_BEGIN = "# ---- prompt + parser: COPIED VERBATIM"
_END = "# ----------"


def _load_windows_block():
    src = _SCRIPT.read_text()
    start = src.index(_BEGIN)
    end = src.index(_END, start)
    block = src[start:end]
    ns: dict = {}
    exec(compile(block, str(_SCRIPT), "exec"), ns)
    return ns


def test_windows_prompt_matches_app():
    ns = _load_windows_block()
    assert ns["build_prompt"]() == canon.build_prompt()
    assert ns["CANONICAL_WORDS"] == canon.CANONICAL_WORDS


def test_windows_parser_matches_app():
    ns = _load_windows_block()
    win = ns["parse_response"]
    cases = ["anger", "Neutral", "The emotion is happiness.", "joyful",
             "The speaker sounds angry", "surprised", "afraid", "calm",
             "I cannot tell", ""]
    for c in cases:
        w = win(c)
        a = canon.parse_response(c)
        assert (w or None) == a, f"parser drift on {c!r}: windows={w!r} app={a!r}"
    # synonym table identical
    assert ns["_SYNONYMS"] == canon._SYNONYMS
