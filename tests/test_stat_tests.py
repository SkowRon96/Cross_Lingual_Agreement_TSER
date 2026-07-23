from app.analysis.tests import mcnemar, two_proportion


def test_mcnemar_symmetric_is_nonsignificant():
    # equal discordant counts -> p ~ 1
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [False] * 10
    r = mcnemar(a, b)
    assert r.p_value > 0.9


def test_mcnemar_detects_asymmetry():
    # A correct where B wrong on many, rarely reverse
    a = [True] * 20 + [False] * 2
    b = [False] * 20 + [True] * 2
    r = mcnemar(a, b)
    assert r.p_value < 0.001


def test_two_proportion_basic():
    r = two_proportion(80, 100, 60, 100)
    assert r.name == "two_proportion"
    assert r.p_value < 0.05
