import math
from app.config.emotions import Emotion
from app.config import studies
from app.models.base import Prediction
from app.analysis.agreement import aligned_pairs, cohen_kappa, fleiss_kappa, agreement

A = Emotion.ANGER
S = Emotion.SADNESS
SUP = Emotion.SURPRISE


def _p(sid, pred, mid="m", abst=False):
    return Prediction(sample_id=sid, model_id=mid, modality="text", pred_raw="x",
                      pred=pred, confidence=1.0 if pred else None,
                      scores={pred: 1.0} if pred else {}, abstained=abst)


def test_cohen_known_value():
    # 8 agree on A, 2 disagree -> classic; verify against hand calc
    a = [A, A, A, A, A, S, S, S, S, S]
    b = [A, A, A, A, S, S, S, S, S, A]
    k = cohen_kappa(a, b)
    # po = 8/10=0.8; pe = (5/10*4/10)+(5/10*6/10)=0.2+0.3=0.5; kappa=(0.8-0.5)/(1-0.5)=0.6
    assert abs(k - 0.6) < 1e-9


def test_common_support_excludes_surprise_for_roberta6():
    gold = {"1": A, "2": SUP, "3": S}
    pa = [_p("1", A), _p("2", A), _p("3", S)]           # roberta6 (no surprise support)
    pb = [_p("1", A), _p("2", S), _p("3", S)]
    la, lb, exc = aligned_pairs(pa, pb, gold,
                                studies.SUPPORTED["pl_text_roberta6"],
                                studies.SUPPORTED["speech_zeroshot"])
    assert len(la) == 2 and len(lb) == 2       # sample "2" (gold=surprise) dropped
    assert exc["gold_out_of_support"] == 1


def test_abstain_excluded():
    gold = {"1": A, "2": S}
    pa = [_p("1", A), _p("2", None, abst=True)]
    pb = [_p("1", A), _p("2", S)]
    la, lb, exc = aligned_pairs(pa, pb, gold, set(Emotion), set(Emotion))
    assert len(la) == 1 and exc["abstain_a"] == 1


def test_fleiss_perfect_agreement():
    r1 = [A, S, A]
    r2 = [A, S, A]
    r3 = [A, S, A]
    assert abs(fleiss_kappa([r1, r2, r3]) - 1.0) < 1e-9


def test_agreement_sets_headline_flag():
    gold = {"1": A, "2": S}
    pa = [_p("1", A, "mling_tabularis"), _p("2", S, "mling_tabularis")]
    pb = [_p("1", A, "speech_zeroshot"), _p("2", S, "speech_zeroshot")]
    pair = frozenset({"mling_tabularis", "speech_zeroshot"})
    r = agreement(pa, pb, gold, "mling_tabularis", "speech_zeroshot", "nEMO",
                  headline_pair=pair)
    assert r.headline is True and r.secondary_flag is False
    assert r.n_used == 2 and abs(r.kappa - 1.0) < 1e-9


def test_wrong_pair_not_headline_even_if_both_headline_models():
    # text-text pair (en_text_hartmann + mling_tabularis) on RAVDESS must NOT be headline
    gold = {"1": A, "2": S}
    pa = [_p("1", A), _p("2", S)]
    pb = [_p("1", A), _p("2", A)]
    pair = frozenset({"en_text_hartmann", "speech_zeroshot"})  # EN corpus headline
    r = agreement(pa, pb, gold, "en_text_hartmann", "mling_tabularis", "RAVDESS",
                  headline_pair=pair)
    assert r.headline is False


def test_cohen_single_class_perfect_agreement_is_one():
    # both raters constant on the same class -> perfect agreement -> kappa 1.0
    assert abs(cohen_kappa([A, A, A], [A, A, A]) - 1.0) < 1e-9
