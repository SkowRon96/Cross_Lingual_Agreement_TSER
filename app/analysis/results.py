"""Pydantic result schemas + deterministic JSON / tidy-CSV serialization."""
from __future__ import annotations
import csv
import json
from pathlib import Path

from pydantic import BaseModel


class BootstrapCI(BaseModel):
    lo: float
    hi: float
    level: float = 0.95


class Baseline(BaseModel):
    model_id: str
    corpus: str
    modality: str
    n: int
    accuracy: float
    macro_f1: float
    confusion: dict[str, dict[str, int]]  # gold -> pred -> count
    labels: list[str]


class AgreementResult(BaseModel):
    corpus: str
    model_a: str
    model_b: str
    kappa: float
    n_used: int
    n_excluded: int
    excluded_reasons: dict[str, int]
    headline: bool = False
    secondary_flag: bool = False  # any of the pair is secondary/assumed
    ci: BootstrapCI | None = None


class FleissResult(BaseModel):
    corpus: str
    model_ids: list[str]
    kappa: float
    n_used: int
    n_excluded: int
    ci: BootstrapCI | None = None


class TestResult(BaseModel):
    name: str            # "mcnemar" | "chi2" | "two_proportion"
    corpus: str
    detail: str          # what was compared, human-readable
    statistic: float
    p_value: float
    extra: dict[str, float] = {}


class ConflictResult(BaseModel):
    corpus: str
    model_a: str
    model_b: str
    n_conflicts: int
    a_closer_to_gold: int
    b_closer_to_gold: int
    both_wrong: int
    per_emotion: dict[str, int]  # gold emotion -> conflict count


class ConfidenceCorr(BaseModel):
    corpus: str
    model_a: str
    model_b: str
    n: int
    pearson: float
    spearman: float


class Study2LangRow(BaseModel):
    split: str
    language: str
    kappa: float
    n_used: int
    ci: BootstrapCI | None = None
    syntactic_distance: float | None = None
    resource_tier: int | None = None


class AnalysisBundle(BaseModel):
    study: int
    baselines: list[Baseline] = []
    agreements: list[AgreementResult] = []
    fleiss: list[FleissResult] = []
    tests: list[TestResult] = []
    conflicts: list[ConflictResult] = []
    confidence: list[ConfidenceCorr] = []
    study2: list[Study2LangRow] = []


def write_json(bundle: AnalysisBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.model_dump(), indent=2, sort_keys=True))


def read_json(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(Path(path).read_text())


def write_tidy_csv(bundle: AnalysisBundle, path: Path) -> None:
    """One long tidy table of κ rows (agreements) for quick spreadsheet inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "corpus", "model_a", "model_b", "kappa",
                    "ci_lo", "ci_hi", "n_used", "n_excluded", "headline", "secondary"])
        for a in bundle.agreements:
            w.writerow([bundle.study, a.corpus, a.model_a, a.model_b, a.kappa,
                        a.ci.lo if a.ci else "", a.ci.hi if a.ci else "",
                        a.n_used, a.n_excluded, a.headline, a.secondary_flag])
