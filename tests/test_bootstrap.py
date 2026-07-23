from app.config.emotions import Emotion
from app.analysis.bootstrap import bootstrap_kappa, bootstrap_diff

A, S = Emotion.ANGER, Emotion.SADNESS


def test_ci_is_deterministic_and_ordered():
    a = [A, A, A, A, A, S, S, S, S, S]
    b = [A, A, A, A, S, S, S, S, S, A]
    ci1 = bootstrap_kappa(a, b, n=200)
    ci2 = bootstrap_kappa(a, b, n=200)
    assert ci1.lo == ci2.lo and ci1.hi == ci2.hi       # deterministic (fixed seed)
    assert ci1.lo <= ci1.hi
    assert -1.0 <= ci1.lo and ci1.hi <= 1.0


def test_perfect_agreement_ci_at_one():
    a = [A, S, A, S, A, S]
    b = [A, S, A, S, A, S]
    ci = bootstrap_kappa(a, b, n=200)
    assert abs(ci.lo - 1.0) < 1e-9 and abs(ci.hi - 1.0) < 1e-9


def test_diff_ci_deterministic():
    a1 = [A, A, S, S]; b1 = [A, A, S, S]
    a2 = [A, S, A, S]; b2 = [S, A, S, A]
    d, ci, p = bootstrap_diff(a1, b1, a2, b2, n=200)
    assert ci.lo <= ci.hi     # CI ordered
    assert 0.0 <= p <= 1.0
