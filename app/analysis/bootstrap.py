"""Percentile bootstrap CIs for kappa (fixed seed for reproducibility)."""
import numpy as np

from app.analysis.agreement import cohen_kappa
from app.analysis.results import BootstrapCI

SEED = 20260723


def _resampled_kappa(la, lb, idx):
    a = [la[i] for i in idx]
    b = [lb[i] for i in idx]
    return cohen_kappa(a, b)


def bootstrap_kappa(labels_a, labels_b, *, n: int = 1000, seed: int = SEED) -> BootstrapCI:
    m = len(labels_a)
    if m == 0:
        return BootstrapCI(lo=0.0, hi=0.0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, size=m)
        vals[k] = _resampled_kappa(labels_a, labels_b, idx)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return BootstrapCI(lo=float(lo), hi=float(hi))


def bootstrap_diff(la1, lb1, la2, lb2, *, n: int = 1000, seed: int = SEED):
    m1, m2 = len(la1), len(la2)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        i1 = rng.integers(0, m1, size=m1)
        i2 = rng.integers(0, m2, size=m2)
        k1 = _resampled_kappa(la1, lb1, i1)
        k2 = _resampled_kappa(la2, lb2, i2)
        diffs[k] = k1 - k2
    point = cohen_kappa(la1, lb1) - cohen_kappa(la2, lb2)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, BootstrapCI(lo=float(lo), hi=float(hi))
