from app.config.emotions import Emotion
from app.config import studies
from app.analysis.results import (
    Baseline, AgreementResult, BootstrapCI, AnalysisBundle, write_json, read_json,
)


def test_study_config_headline_and_support():
    assert studies.HEADLINE[("PL", "text")] == "mling_tabularis"
    assert studies.HEADLINE[("EN", "speech")] == "speech_zeroshot"
    # roberta6/poltextlab lack surprise
    assert Emotion.SURPRISE not in studies.SUPPORTED["xlmr_pooled10"]
    assert Emotion.SURPRISE not in studies.SUPPORTED["pl_text_roberta6"]
    # iemocap 4-class
    assert studies.SUPPORTED["en_speech_iemocap"] == {
        Emotion.ANGER, Emotion.HAPPINESS, Emotion.SADNESS, Emotion.NEUTRAL}
    # every study-2 split maps to a language
    for s in studies.STUDY2_SPLITS:
        assert s in studies.SPLIT_LANG


def test_bundle_roundtrip(tmp_path):
    b = AnalysisBundle(
        study=1,
        baselines=[Baseline(model_id="m", corpus="nEMO", modality="text",
                            n=10, accuracy=0.5, macro_f1=0.4,
                            confusion={"anger": {"anger": 3}}, labels=["anger"])],
        agreements=[AgreementResult(corpus="nEMO", model_a="a", model_b="b",
                    kappa=0.42, n_used=8, n_excluded=2,
                    excluded_reasons={"gold_out_of_support": 1, "abstain": 1},
                    headline=True, ci=BootstrapCI(lo=0.1, hi=0.7))],
    )
    p = tmp_path / "study1.json"
    write_json(b, p)
    got = read_json(p)
    assert got.study == 1
    assert got.agreements[0].kappa == 0.42
    assert got.agreements[0].ci.hi == 0.7
