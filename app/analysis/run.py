"""analyze orchestration: read cached predictions + gold, emit the AnalysisBundle.

Re-runnable and tolerant of missing artefacts: a corpus with no ingested index, or a model
with no stored predictions, is skipped with a warning (never fabricated). Run once early to
exercise the pipeline, then again after `import-predictions` brings in the Qwen speech labels.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

from app.analysis import bootstrap, conflicts, correlation, lang_features, metrics
from app.analysis.agreement import agreement, aligned_pairs, cohen_kappa, fleiss_kappa
from app.analysis.results import (
    AnalysisBundle, BootstrapCI, FleissResult, Study2LangRow)
from app.analysis.tests import correctness, mcnemar, two_proportion
from app.config import studies
from app.config.emotions import Emotion
from app.labels.mapping import map_label
from app.predict.store import PredictionStore


def _warn(msg: str) -> None:
    print(f"[analyze] WARNING: {msg}", file=sys.stderr)


def gold_lookup(study: int, corpus: str, index_dir: Path) -> dict[str, Emotion]:
    import pyarrow.parquet as pq
    path = Path(index_dir) / f"{corpus}.parquet"
    cols = pq.read_table(path, columns=["sample_id", "gold_raw"]).to_pydict()
    out: dict[str, Emotion] = {}
    for sid, raw in zip(cols["sample_id"], cols["gold_raw"]):
        mapped = map_label(raw)          # canonical Emotion, or None for a dropped label
        if mapped is not None:
            out[sid] = mapped
    return out


def _load(store: PredictionStore, study: int, corpus: str, model_id: str):
    if not store.path(study, corpus, model_id).exists():
        return None
    return store.read(study, corpus, model_id)


def _bootstrap_fleiss(rater_labels, *, n: int = 1000, seed: int = bootstrap.SEED) -> BootstrapCI:
    m = len(rater_labels[0])
    if m == 0:
        return BootstrapCI(lo=0.0, hi=0.0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, size=m)
        resampled = [[r[i] for i in idx] for r in rater_labels]
        vals[k] = fleiss_kappa(resampled)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return BootstrapCI(lo=float(lo), hi=float(hi))


def _fleiss_ravdess(bundle: AnalysisBundle, present: dict, gold: dict, trio: list[str]) -> None:
    maps = {m: {p.sample_id: p for p in present[m]} for m in trio}
    support = set(Emotion)
    for m in trio:
        support = support & studies.SUPPORTED[m]
    rater_labels: list[list[Emotion]] = [[] for _ in trio]
    n_used = n_excluded = 0
    for sid, g in gold.items():
        preds = [maps[m].get(sid) for m in trio]
        if any(p is None or p.pred is None for p in preds) or g not in support:
            n_excluded += 1
            continue
        for r, p in enumerate(preds):
            rater_labels[r].append(p.pred)
        n_used += 1
    if n_used == 0:
        return
    k = fleiss_kappa(rater_labels)
    ci = _bootstrap_fleiss(rater_labels)
    bundle.fleiss.append(FleissResult(
        corpus="RAVDESS", model_ids=trio, kappa=k,
        n_used=n_used, n_excluded=n_excluded, ci=ci))


def _study2(bundle: AnalysisBundle, store: PredictionStore, index_dir: Path) -> None:
    text_m = studies.HEADLINE[("PL", "text")]    # mling_tabularis (multilingual)
    speech_m = studies.HEADLINE[("PL", "speech")]  # speech_zeroshot
    pl_acc = None  # (k_correct, n) for the PL split, for PL-vs-others two-proportion
    per_lang_acc: list[tuple[str, str, int, int]] = []  # (split, iso, k, n)
    for split in studies.STUDY2_SPLITS:
        idx_path = Path(index_dir) / f"{split}.parquet"
        if not idx_path.exists():
            _warn(f"study 2: no index for split {split!r}, skipping")
            continue
        gold = gold_lookup(2, split, index_dir)
        pt = _load(store, 2, split, text_m)
        ps = _load(store, 2, split, speech_m)
        iso = studies.SPLIT_LANG[split]
        if pt is not None and ps is not None:
            la, lb, _ = aligned_pairs(pt, ps, gold,
                                      studies.SUPPORTED[text_m], studies.SUPPORTED[speech_m])
            k = cohen_kappa(la, lb)
            ci = bootstrap.bootstrap_kappa(la, lb) if la else None
            bundle.study2.append(Study2LangRow(
                split=split, language=iso, kappa=k, n_used=len(la), ci=ci,
                syntactic_distance=lang_features.SYNTACTIC_DISTANCE.get(iso),
                resource_tier=lang_features.RESOURCE_TIER.get(iso)))
        # accuracy of the headline text model per language (descriptive PL-vs-others)
        if pt is not None:
            corr = correctness(pt, gold)
            k_ok = sum(1 for v in corr.values() if v)
            per_lang_acc.append((split, iso, k_ok, len(corr)))
            if iso == "pl":
                pl_acc = (k_ok, len(corr))
    if pl_acc and pl_acc[1] > 0:
        for split, iso, k_ok, n in per_lang_acc:
            if iso == "pl" or n == 0:
                continue
            bundle.tests.append(two_proportion(
                pl_acc[0], pl_acc[1], k_ok, n, corpus=split,
                detail=f"{text_m} accuracy: pl vs {iso} ({split})"))


def analyze_study(study, *, results_root: Path, index_dir: Path) -> AnalysisBundle:
    store = PredictionStore(Path(results_root) / "predictions")
    index_dir = Path(index_dir)
    bundle = AnalysisBundle(study=study)
    corpora = [c for (s, c) in studies.ROSTER if s == study]
    headline_aligned: dict[str, tuple[list, list]] = {}  # lang -> (la, lb) for PL-vs-EN diff
    for corpus in corpora:
        if study == 2:
            continue  # study 2 handled corpus-wise in _study2
        idx_path = index_dir / f"{corpus}.parquet"
        if not idx_path.exists():
            _warn(f"no index for corpus {corpus!r} ({idx_path}); skipping")
            continue
        gold = gold_lookup(study, corpus, index_dir)
        present: dict[str, list] = {}
        for mid in studies.ROSTER[(study, corpus)]:
            preds = _load(store, study, corpus, mid)
            if preds is None:
                _warn(f"no predictions for {mid} on {corpus} (study {study}); skipping")
                continue
            present[mid] = preds
            bundle.baselines.append(
                metrics.baseline(preds, gold, mid, corpus, studies.MODALITY[mid]))
        lang = "PL" if corpus == "nEMO" else ("EN" if corpus == "RAVDESS" else None)
        headline_pair = None
        if lang:
            headline_pair = frozenset({studies.HEADLINE[(lang, "text")],
                                       studies.HEADLINE[(lang, "speech")]})
        mids = list(present)
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                a, b = mids[i], mids[j]
                res = agreement(present[a], present[b], gold, a, b, corpus,
                                headline_pair=headline_pair)
                la, lb, _ = aligned_pairs(present[a], present[b], gold,
                                          studies.SUPPORTED[a], studies.SUPPORTED[b])
                res.ci = bootstrap.bootstrap_kappa(la, lb) if la else None
                bundle.agreements.append(res)
        if lang:
            ht, hs = studies.HEADLINE[(lang, "text")], studies.HEADLINE[(lang, "speech")]
            if ht in present and hs in present:
                hla, hlb, _ = aligned_pairs(present[ht], present[hs], gold,
                                            studies.SUPPORTED[ht], studies.SUPPORTED[hs])
                if hla:
                    headline_aligned[lang] = (hla, hlb)
                bundle.conflicts.append(
                    conflicts.conflict(present[ht], present[hs], gold, ht, hs, corpus))
                cc = correlation.confidence_corr(present[ht], present[hs], ht, hs, corpus)
                if cc:
                    bundle.confidence.append(cc)
                ca = correctness(present[ht], gold)
                cb = correctness(present[hs], gold)
                shared = [sid for sid in ca if sid in cb]
                if shared:
                    bundle.tests.append(mcnemar(
                        [ca[s] for s in shared], [cb[s] for s in shared],
                        corpus=corpus, detail=f"{ht} vs {hs} correctness"))
        if corpus == "RAVDESS":
            trio = ["en_text_hartmann", "speech_zeroshot", "en_face_vit"]
            if all(m in present for m in trio):
                _fleiss_ravdess(bundle, present, gold, trio)
    # PL vs EN headline κ difference (study 1, unpaired: independent resamples)
    if "PL" in headline_aligned and "EN" in headline_aligned:
        (pla, plb), (ela, elb) = headline_aligned["PL"], headline_aligned["EN"]
        point, ci, p = bootstrap.bootstrap_diff(pla, plb, ela, elb)
        bundle.tests.append(TestResult(
            name="kappa_diff_pl_vs_en", corpus="nEMO|RAVDESS",
            detail="headline text↔speech κ: PL (nEMO) minus EN (RAVDESS)",
            statistic=point, p_value=p, extra={"ci_lo": ci.lo, "ci_hi": ci.hi}))
    if study == 2:
        _study2(bundle, store, index_dir)
    return bundle
