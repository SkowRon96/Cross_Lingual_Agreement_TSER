from app.analysis.results import Baseline, AnalysisBundle, AgreementResult, BootstrapCI, ConflictResult, Study2LangRow
from app.report import figures, style


def test_confusion_fig_saves_png(tmp_path):
    b = Baseline(model_id="m", corpus="nEMO", modality="text", n=3, accuracy=0.6,
                 macro_f1=0.5, confusion={"anger": {"anger": 2, "sadness": 1}},
                 labels=["anger", "sadness"])
    fig = figures.confusion_fig(b)
    style.save_fig(fig, tmp_path / "cm")
    assert (tmp_path / "cm.png").exists() and not (tmp_path / "cm.pdf").exists()


def test_kappa_ci_and_conflict_and_scatter(tmp_path):
    bundle = AnalysisBundle(
        study=2,
        agreements=[AgreementResult(corpus="nEMO", model_a="a", model_b="b", kappa=0.4,
                    n_used=5, n_excluded=0, excluded_reasons={}, headline=True,
                    ci=BootstrapCI(lo=0.1, hi=0.7))],
        conflicts=[ConflictResult(corpus="nEMO", model_a="a", model_b="b", n_conflicts=3,
                   a_closer_to_gold=1, b_closer_to_gold=1, both_wrong=1,
                   per_emotion={"anger": 2, "sadness": 1})],
        study2=[Study2LangRow(split="nemo", language="pl", kappa=0.5, n_used=10,
                syntactic_distance=0.0, resource_tier=4),
                Study2LangRow(split="cafe", language="fr", kappa=0.3, n_used=8,
                syntactic_distance=0.55, resource_tier=5)])
    style.save_fig(figures.kappa_ci_fig(bundle), tmp_path / "k")
    style.save_fig(figures.conflict_fig(bundle.conflicts[0]), tmp_path / "c")
    style.save_fig(figures.study2_scatter(bundle), tmp_path / "s")
    for name in ("k", "c", "s"):
        assert (tmp_path / f"{name}.png").exists()
