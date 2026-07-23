from app.analysis import lang_features as lf
from app.config import studies


def test_every_language_has_features():
    langs = set(studies.SPLIT_LANG.values())
    for iso in langs:
        assert iso in lf.SYNTACTIC_DISTANCE, iso
        assert iso in lf.RESOURCE_TIER, iso


def test_polish_distance_is_zero():
    assert lf.SYNTACTIC_DISTANCE["pl"] == 0.0


def test_analyze_runs_with_no_data(tmp_path):
    """analyze must not crash when no index/predictions exist yet (partial early run)."""
    from app.analysis.run import analyze_study
    bundle = analyze_study(1, results_root=tmp_path, index_dir=tmp_path / "index")
    assert bundle.study == 1
    assert bundle.baselines == [] and bundle.agreements == []
