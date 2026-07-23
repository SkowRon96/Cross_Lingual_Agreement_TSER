from app.analysis.results import AnalysisBundle, AgreementResult, BootstrapCI, Baseline
from app.report import tables


def _bundle():
    return AnalysisBundle(
        study=1,
        baselines=[Baseline(model_id="mling_tabularis", corpus="nEMO", modality="text",
                            n=10, accuracy=0.6, macro_f1=0.5, confusion={}, labels=[])],
        agreements=[AgreementResult(corpus="nEMO", model_a="mling_tabularis",
                    model_b="speech_zeroshot", kappa=0.42, n_used=8, n_excluded=2,
                    excluded_reasons={}, headline=True, ci=BootstrapCI(lo=0.1, hi=0.7))])


def test_kappa_table_has_ci_and_flags():
    df = tables.kappa_table(_bundle())
    assert "kappa" in df.columns and "ci_lo" in df.columns
    assert df.iloc[0]["headline"] in (True, "True", 1)


def test_write_table_emits_four_formats(tmp_path):
    df = tables.kappa_table(_bundle())
    stem = tmp_path / "kappa"
    tables.write_table(df, stem)
    for ext in ("csv", "md", "tex", "png"):
        assert (tmp_path / f"kappa.{ext}").exists()
