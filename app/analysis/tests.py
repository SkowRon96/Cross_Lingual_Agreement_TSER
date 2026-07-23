"""Significance tests: McNemar (paired within-set), two-proportion / chi-square (PL vs EN)."""
import math

from scipy import stats

from app.analysis.results import TestResult
from app.config.emotions import Emotion
from app.models.base import Prediction


def correctness(preds: list[Prediction], gold: dict[str, Emotion]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for p in preds:
        if p.sample_id in gold:
            out[p.sample_id] = (p.pred is not None and p.pred == gold[p.sample_id])
    return out


def mcnemar(correct_a: list[bool], correct_b: list[bool], corpus: str = "",
            detail: str = "") -> TestResult:
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    disc = b + c
    if disc == 0:
        return TestResult(name="mcnemar", corpus=corpus, detail=detail,
                          statistic=0.0, p_value=1.0, extra={"b": b, "c": c})
    if disc < 25:
        p = 2 * sum(math.comb(disc, i) * 0.5 ** disc for i in range(0, min(b, c) + 1))
        p = min(1.0, p)
        stat = float(min(b, c))
    else:
        stat = (abs(b - c) - 1) ** 2 / disc
        p = 1 - stats.chi2.cdf(stat, df=1)
    return TestResult(name="mcnemar", corpus=corpus, detail=detail,
                      statistic=float(stat), p_value=float(p), extra={"b": b, "c": c})


def two_proportion(k1: int, n1: int, k2: int, n2: int, corpus: str = "",
                   detail: str = "") -> TestResult:
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return TestResult(name="two_proportion", corpus=corpus, detail=detail,
                      statistic=float(z), p_value=float(pval),
                      extra={"p1": p1, "p2": p2})
