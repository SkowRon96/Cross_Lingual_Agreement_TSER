"""Study 3 (arbitration + abstention) on synthetic data with known answers."""
import numpy as np
import pytest

from app.analysis import arbitration as arb
from app.config.emotions import CANONICAL_ORDER, Emotion
from app.models.base import Prediction
from app.predict.store import PredictionStore

STUDY = 1
CORPUS = "SYNTH"


def _pred(sample_id: str, model_id: str, modality: str, scores: dict[Emotion, float],
          *, abstain: bool = False) -> Prediction:
    if abstain:
        return Prediction(sample_id=sample_id, model_id=model_id, modality=modality,
                          pred_raw=None, pred=None, confidence=None, scores={},
                          abstained=True, abstain_reason="synthetic")
    pred = max(scores, key=lambda e: (scores[e], -CANONICAL_ORDER.index(e)))
    return Prediction(sample_id=sample_id, model_id=model_id, modality=modality,
                      pred_raw=pred.value, pred=pred, confidence=scores[pred],
                      scores=scores, abstained=False)


def _onehot(e: Emotion) -> dict[Emotion, float]:
    return {e: 1.0}


def _soft(top: Emotion, p: float) -> dict[Emotion, float]:
    rest = [e for e in CANONICAL_ORDER if e != top]
    return {top: p, **{e: (1.0 - p) / len(rest) for e in rest}}


def _store(tmp_path, channels: dict[str, list[Prediction]]) -> PredictionStore:
    store = PredictionStore(tmp_path / "predictions")
    for model_id, preds in channels.items():
        store.write(STUDY, CORPUS, model_id, preds, config_hash="synthetic")
    return store


# ------------------------------------------------------------------ vector arithmetic
def test_fuse_sums_harmonised_vectors_with_weights():
    row = arb.SampleRow(
        sample_id="s", gold=Emotion.ANGER,
        preds={"text": Emotion.ANGER, "speech": Emotion.SADNESS},
        scores={"text": {Emotion.ANGER: 0.6, Emotion.SADNESS: 0.4},
                "speech": {Emotion.SADNESS: 1.0}},
    )
    fused = arb.fuse(row, {"text": 1.0, "speech": 1.0})
    assert fused[Emotion.ANGER] == pytest.approx(0.6)
    assert fused[Emotion.SADNESS] == pytest.approx(1.4)
    assert arb.argmax_vector(fused) is Emotion.SADNESS
    # weights rescale the contributions, not the class space
    weighted = arb.fuse(row, {"text": 0.8, "speech": 0.2})
    assert weighted[Emotion.ANGER] == pytest.approx(0.48)
    assert weighted[Emotion.SADNESS] == pytest.approx(0.52)
    assert arb.argmax_vector(weighted) is Emotion.SADNESS
    # a channel with weight 0 contributes nothing at all
    muted = arb.fuse(row, {"text": 1.0, "speech": 0.0})
    assert muted[Emotion.SADNESS] == pytest.approx(0.4)


def test_argmax_tie_break_is_canonical_order_not_dict_order():
    vec = {Emotion.NEUTRAL: 0.5, Emotion.ANGER: 0.5}
    assert arb.argmax_vector(vec) is Emotion.ANGER  # ANGER precedes NEUTRAL canonically
    assert arb.argmax_vector({e: 0.0 for e in CANONICAL_ORDER}) is Emotion.ANGER


def test_degenerate_onehot_channel_outvotes_a_diffuse_channel(tmp_path):
    """Qwen-style one-hot speech: mass 1.0 on one class beats any non-saturated vector."""
    gold = {}
    text, speech = [], []
    for i in range(20):
        sid = f"s{i:02d}"
        gold[sid] = Emotion.ANGER
        text.append(_pred(sid, "t", "text", _soft(Emotion.ANGER, 0.95)))
        speech.append(_pred(sid, "a", "speech", _onehot(Emotion.SADNESS)))
    store = _store(tmp_path, {"t": text, "a": speech})
    rows, _ = arb.build_table(store, CORPUS, {"text": "t", "speech": "a"}, gold)
    fused = [arb.argmax_vector(arb.fuse(r, {"text": 1.0, "speech": 1.0})) for r in rows]
    assert all(p is Emotion.SADNESS for p in fused)     # 1.0 > 0.95
    specs_degenerate = all(max(r.scores["speech"].values()) >= 0.999 for r in rows)
    assert specs_degenerate


def test_build_table_drops_abstained_and_missing_channels(tmp_path):
    gold = {f"s{i}": Emotion.ANGER for i in range(4)}
    text = [_pred(f"s{i}", "t", "text", _soft(Emotion.ANGER, 0.9)) for i in range(4)]
    speech = [_pred("s0", "a", "speech", _onehot(Emotion.ANGER)),
              _pred("s1", "a", "speech", {}, abstain=True),
              _pred("s2", "a", "speech", _onehot(Emotion.FEAR))]
    # s3 has no speech row at all
    store = _store(tmp_path, {"t": text, "a": speech})
    rows, excluded = arb.build_table(store, CORPUS, {"text": "t", "speech": "a"}, gold)
    assert [r.sample_id for r in rows] == ["s0", "s2"]
    assert excluded == {"abstained:speech": 1, "missing_prediction:speech": 1}


def test_build_table_rejects_unnormalised_vectors(tmp_path):
    gold = {"s0": Emotion.ANGER}
    bad = Prediction(sample_id="s0", model_id="t", modality="text", pred_raw="anger",
                     pred=Emotion.ANGER, confidence=2.0,
                     scores={Emotion.ANGER: 2.0, Emotion.FEAR: 1.0})
    store = _store(tmp_path, {"t": [bad]})
    with pytest.raises(ValueError, match="sums to"):
        arb.build_table(store, CORPUS, {"text": "t"}, gold)


# ------------------------------------------------------------------------- weights / S2
def test_fit_weights_uses_conflict_rows_only():
    def row(sid, gold, t, a):
        return arb.SampleRow(sample_id=sid, gold=gold, preds={"text": t, "speech": a},
                             scores={"text": _onehot(t), "speech": _onehot(a)})
    rows = [
        # agreement rows: both right, they carry no arbitration information
        row("a1", Emotion.ANGER, Emotion.ANGER, Emotion.ANGER),
        row("a2", Emotion.ANGER, Emotion.ANGER, Emotion.ANGER),
        # conflict rows: speech right 3/4, text right 1/4
        row("c1", Emotion.FEAR, Emotion.FEAR, Emotion.SADNESS),
        row("c2", Emotion.FEAR, Emotion.SADNESS, Emotion.FEAR),
        row("c3", Emotion.FEAR, Emotion.SADNESS, Emotion.FEAR),
        row("c4", Emotion.FEAR, Emotion.SADNESS, Emotion.FEAR),
    ]
    w, n_conf = arb.fit_weights(rows, ["text", "speech"])
    assert n_conf == 4
    assert w["text"] == pytest.approx(0.25)    # 0.25 / (0.25 + 0.75)
    assert w["speech"] == pytest.approx(0.75)
    assert sum(w.values()) == pytest.approx(1.0)


def test_s2_weights_never_see_the_evaluated_fold(tmp_path, monkeypatch):
    """Trap #1: out-of-fold fitting. Fold k's weights must not touch fold k's samples."""
    rng = np.random.default_rng(7)
    n, gold, text, speech = 60, {}, [], []
    for i in range(n):
        sid = f"s{i:03d}"
        g = CANONICAL_ORDER[int(rng.integers(0, 7))]
        gold[sid] = g
        text.append(_pred(sid, "t", "text", _soft(CANONICAL_ORDER[int(rng.integers(0, 7))], 0.7)))
        speech.append(_pred(sid, "a", "speech", _onehot(CANONICAL_ORDER[int(rng.integers(0, 7))])))
    store = _store(tmp_path, {"t": text, "a": speech})
    seen: list[set[str]] = []
    real_fit = arb.fit_weights
    monkeypatch.setattr(arb, "fit_weights",
                        lambda rows, mods: (seen.append({r.sample_id for r in rows}),
                                            real_fit(rows, mods))[1])
    res = arb.arbitrate_corpus(store, CORPUS, gold, variant="synthetic",
                               channels={"text": "t", "speech": "a"}, n_folds=5)
    assert len(res.fold_weights) == 5
    folds = arb.fold_assignment(n, 5, res.seed)
    ids = sorted(gold)
    for f in range(5):
        eval_ids = {ids[i] for i in range(n) if folds[i] == f}
        assert seen[f].isdisjoint(eval_ids), f"fold {f}: fitting rows leaked into evaluation"
        assert len(seen[f]) == n - len(eval_ids)
    assert sum(fw.n_eval for fw in res.fold_weights) == n


def test_fold_assignment_is_deterministic_and_balanced():
    a = arb.fold_assignment(37, 5, seed=123)
    b = arb.fold_assignment(37, 5, seed=123)
    assert (a == b).all()
    counts = np.bincount(a, minlength=5)
    assert counts.sum() == 37 and counts.max() - counts.min() <= 1
    assert not (arb.fold_assignment(37, 5, seed=124) == a).all()


# ------------------------------------------------------------------------ oracle ceiling
def _random_corpus(tmp_path, n=240, seed=11, with_face=True):
    rng = np.random.default_rng(seed)
    gold, text, speech, face = {}, [], [], []
    for i in range(n):
        sid = f"s{i:03d}"
        g = CANONICAL_ORDER[int(rng.integers(0, 7))]
        gold[sid] = g
        # each channel is right with a different probability -> partial complementarity
        def draw(p):
            return g if rng.random() < p else CANONICAL_ORDER[int(rng.integers(0, 7))]
        text.append(_pred(sid, "t", "text", _soft(draw(0.45), float(rng.uniform(0.3, 0.9)))))
        speech.append(_pred(sid, "a", "speech", _onehot(draw(0.55))))
        face.append(_pred(sid, "v", "face", _soft(draw(0.35), float(rng.uniform(0.3, 0.9)))))
    channels = {"text": "t", "speech": "a"}
    payload = {"t": text, "a": speech}
    if with_face:
        channels["face"] = "v"
        payload["v"] = face
    return _store(tmp_path, payload), gold, channels


@pytest.mark.parametrize("with_face", [False, True])
def test_oracle_is_the_ceiling_of_late_fusion(tmp_path, with_face):
    store, gold, channels = _random_corpus(tmp_path, with_face=with_face)
    res = arb.arbitrate_corpus(store, CORPUS, gold, variant="synthetic", channels=channels)
    by = {s.strategy: s for s in res.strategies}
    for key in ("S0", "S1", "S2", "S2_insample"):
        assert by["S3"].accuracy >= by[key].accuracy - 1e-12
    # the oracle accuracy is exactly "some channel was right"
    rows, _ = arb.build_table(store, CORPUS, channels, gold)
    expected = sum(1 for r in rows if any(r.preds[m] == r.gold for m in channels)) / len(rows)
    assert by["S3"].accuracy == pytest.approx(expected)
    assert by["S0"].accuracy == pytest.approx(
        max(c.accuracy for c in res.channels))


def test_oracle_gate_raises_when_violated(tmp_path, monkeypatch):
    """The ceiling check is a hard gate, not a logged warning."""
    store, gold, channels = _random_corpus(tmp_path)
    real = arb._score

    def broken(preds, rows, corpus, name):
        acc, f1 = real(preds, rows, corpus, name)
        return (0.0, 0.0) if name == "S3" else (acc, f1)

    monkeypatch.setattr(arb, "_score", broken)
    with pytest.raises(AssertionError, match="oracle ceiling violated"):
        arb.arbitrate_corpus(store, CORPUS, gold, variant="synthetic", channels=channels)


def test_two_channel_corpus_is_flagged_as_having_no_majority_vote(tmp_path):
    store, gold, channels = _random_corpus(tmp_path, with_face=False)
    res = arb.arbitrate_corpus(store, CORPUS, gold, variant="synthetic", channels=channels)
    assert any("no majority" in note for note in res.notes)
    assert any(c.degenerate and c.modality == "speech" for c in res.channels)


# --------------------------------------------------------------------- risk-coverage
def test_risk_coverage_on_a_hand_computed_case():
    # accept order: two correct, then one wrong, then one correct
    correct = [True, True, False, True]
    curve, aurc = arb.risk_coverage([0, 1, 2, 3], correct)
    assert [c[0] for c in curve] == [0.25, 0.5, 0.75, 1.0]
    assert [c[1] for c in curve] == pytest.approx([0.0, 0.0, 1 / 3, 0.25])
    assert aurc == pytest.approx((0.0 + 0.0 + 1 / 3 + 0.25) / 4)
    assert arb.selective_accuracy(curve, 1.0) == pytest.approx(0.75)
    assert arb.selective_accuracy(curve, 0.5) == pytest.approx(1.0)


def test_risk_coverage_perfect_ranking_has_zero_aurc_prefix():
    correct = [True] * 5 + [False] * 5
    curve, aurc = arb.risk_coverage(list(range(10)), correct)
    assert curve[4][1] == 0.0
    assert curve[-1][1] == pytest.approx(0.5)
    # a perfect ranker puts every error last -> AURC is the mean of the ramp
    expected = np.mean([0, 0, 0, 0, 0, 1 / 6, 2 / 7, 3 / 8, 4 / 9, 5 / 10])
    assert aurc == pytest.approx(expected)


def test_abstention_prefers_the_agreeing_subset(tmp_path):
    """Samples where the channels agree must be accepted first, by construction."""
    gold, text, speech = {}, [], []
    for i in range(40):
        sid = f"s{i:03d}"
        g = Emotion.ANGER if i % 2 == 0 else Emotion.FEAR
        gold[sid] = g
        agree = i < 20
        text.append(_pred(sid, "t", "text", _soft(g, 0.8)))
        speech.append(_pred(sid, "a", "speech", _onehot(g if agree else Emotion.SADNESS)))
    store = _store(tmp_path, {"t": text, "a": speech})
    res = arb.arbitrate_corpus(store, CORPUS, gold, variant="synthetic",
                               channels={"text": "t", "speech": "a"})
    sel = {s.strategy: s for s in res.selective}["S1"]
    assert sel.n_agree == 20 and sel.n_conflict == 20
    assert sel.accuracy_agree_subset == pytest.approx(1.0)
    assert sel.accuracy_agree_subset > sel.accuracy_full_coverage
    # the first half of the accept order is error-free -> risk at 50% coverage is 0
    assert sel.selective_accuracy["1.00"] == pytest.approx(sel.accuracy_full_coverage)
    curve = {round(p.coverage, 4): p.risk for p in sel.curve}
    assert curve[0.5] == pytest.approx(0.0)


# ------------------------------------------------------------------------ serialization
def test_bundle_round_trips_deterministically(tmp_path):
    store, gold, channels = _random_corpus(tmp_path)
    res = arb.arbitrate_corpus(store, CORPUS, gold, variant="synthetic", channels=channels)
    bundle = arb.ArbitrationBundle(corpora=[res])
    out = tmp_path / "analysis" / "study3" / "bundle.json"
    arb.write_json(bundle, out)
    again = arb.read_json(out)
    assert again.model_dump() == bundle.model_dump()
    first = out.read_text()
    arb.write_json(bundle, out)
    assert out.read_text() == first
    arb.write_tidy_md(bundle, tmp_path / "strategies.md")
    md = (tmp_path / "strategies.md").read_text()
    assert "S3" in md and "S2_insample" in md


def test_channel_sets_cover_every_study1_corpus():
    for variant in arb.FULL_VARIANTS:
        per_corpus = arb.CHANNEL_SETS[variant]
        assert set(per_corpus) == {"nEMO", "RAVDESS", "MELD"}
        assert "face" not in per_corpus["nEMO"]     # nEMO has no visual channel


def test_every_channel_is_a_registered_study1_model():
    from app.config import studies
    for variant, per_corpus in arb.CHANNEL_SETS.items():
        for corpus, channels in per_corpus.items():
            for modality, model_id in channels.items():
                assert studies.MODALITY[model_id] == modality, (variant, corpus, model_id)
                assert model_id in studies.ROSTER[(1, corpus)], (variant, corpus, model_id)


# --------------------------------------------------------------- post-audit text variants
def test_audit_variants_swap_only_the_text_channel():
    """D-I / trap #5: identical corpus, identical speech channel, one seed per variant."""
    base = arb.CHANNEL_SETS[arb.AUDIT_BASELINE_VARIANT][arb.AUDIT_CORPUS]
    assert arb.AUDIT_VARIANTS == ["audit_s7", "audit_s17", "audit_s27"]
    for seed, variant in zip(arb.AUDIT_SEEDS, arb.AUDIT_VARIANTS):
        per_corpus = arb.CHANNEL_SETS[variant]
        assert set(per_corpus) == {arb.AUDIT_CORPUS}        # nEMO only
        channels = per_corpus[arb.AUDIT_CORPUS]
        assert set(channels) == set(base)                   # same modalities as the baseline
        assert channels["speech"] == base["speech"]         # speech channel untouched
        assert channels["text"] == f"pl_text_roberta_audit_s{seed}"
        assert channels["text"] != base["text"]


def test_pair_complementarity_counts_and_overlap_ratio():
    g = Emotion.ANGER

    def row(sid, t, a):
        return arb.SampleRow(sample_id=sid, gold=g, preds={"text": t, "speech": a},
                             scores={"text": _onehot(t), "speech": _onehot(a)})
    rows = ([row(f"b{i}", g, g) for i in range(3)]                       # both correct
            + [row(f"t{i}", g, Emotion.FEAR) for i in range(3)]          # text only
            + [row(f"a{i}", Emotion.FEAR, g) for i in range(2)]          # speech only
            + [row(f"n{i}", Emotion.FEAR, Emotion.SADNESS) for i in range(2)])  # both wrong
    c = arb.pair_complementarity(rows, "text", "speech",
                                 {"text": "t", "speech": "a"}, corpus="X", variant="v")
    assert (c.n, c.both_correct, c.a_only_correct, c.b_only_correct, c.both_wrong) == (10, 3, 3, 2, 2)
    assert c.accuracy_a == pytest.approx(0.6) and c.accuracy_b == pytest.approx(0.5)
    assert c.pair_oracle_accuracy == pytest.approx(0.8)
    # errors: 0.4 * 0.5 = 0.2 expected, 0.2 observed -> exactly independent
    assert c.expected_both_wrong_if_independent == pytest.approx(0.2)
    assert c.error_overlap_ratio == pytest.approx(1.0)
    assert -1.0 <= c.kappa_predictions <= 1.0


def test_error_overlap_ratio_detects_coinciding_and_complementary_errors():
    g = Emotion.ANGER

    def rows_for(pairs):
        return [arb.SampleRow(sample_id=f"s{i}", gold=g, preds={"text": t, "speech": a},
                              scores={"text": _onehot(t), "speech": _onehot(a)})
                for i, (t, a) in enumerate(pairs)]
    ch = {"text": "t", "speech": "a"}
    w = Emotion.FEAR
    # identical error pattern -> errors coincide -> ratio > 1
    same = rows_for([(g, g)] * 5 + [(w, w)] * 5)
    assert arb.pair_complementarity(same, "text", "speech", ch, corpus="X",
                                    variant="v").error_overlap_ratio > 1.0
    # disjoint error pattern -> perfectly complementary -> ratio 0
    disjoint = rows_for([(g, w)] * 5 + [(w, g)] * 5)
    assert arb.pair_complementarity(disjoint, "text", "speech", ch, corpus="X",
                                    variant="v").error_overlap_ratio == pytest.approx(0.0)


def test_seed_aggregation_reports_mean_sd_and_delta_vs_baseline(tmp_path):
    store, gold, channels = _random_corpus(tmp_path, with_face=False)
    bundle = arb.ArbitrationBundle()
    for variant in ("base", "seed_a", "seed_b", "seed_c"):
        # different fold seeds -> genuinely different S2 numbers to aggregate over
        offset = {"base": 0, "seed_a": 1, "seed_b": 2, "seed_c": 3}[variant]
        bundle.corpora.append(arb.arbitrate_corpus(
            store, CORPUS, gold, variant=variant, channels=channels, seed=1000 + offset))
    seeds = ["seed_a", "seed_b", "seed_c"]
    deltas, aggs = arb.audit_comparison(bundle, corpus=CORPUS, variants=seeds,
                                        baseline_variant="base", group="test")
    agg = {a.metric: a for a in aggs}
    assert agg["S2_accuracy"].n_seeds == 3
    vals = [agg["S2_accuracy"].per_variant[v] for v in seeds]
    assert agg["S2_accuracy"].mean == pytest.approx(float(np.mean(vals)))
    assert agg["S2_accuracy"].sd == pytest.approx(float(np.std(vals, ddof=1)))
    base_val = {b.variant: b for b in bundle.corpora}["base"]
    base_acc = {s.strategy: s for s in base_val.strategies}["S2"].accuracy
    assert agg["S2_accuracy"].baseline_value == pytest.approx(base_acc)
    assert agg["S2_accuracy"].mean_delta == pytest.approx(float(np.mean(vals)) - base_acc)
    # one delta row per seed x metric, and each delta is value - baseline
    assert {d.variant for d in deltas} == set(seeds)
    for d in deltas:
        assert d.delta == pytest.approx(d.value - d.baseline_value)
    # the section-5.1 answer needs these specific metrics to exist
    for key in ("channel_text_accuracy", "kappa_text_speech", "oracle_gap_S3_minus_S2",
                "error_overlap_ratio_text_speech", "pair_oracle_text_speech"):
        assert key in agg and agg[key].mean_delta is not None


def test_audit_comparison_is_empty_without_the_audit_blocks():
    assert arb.audit_comparison(arb.ArbitrationBundle()) == ([], [])


def test_corpus_metrics_matches_the_underlying_block(tmp_path):
    store, gold, channels = _random_corpus(tmp_path)
    block = arb.arbitrate_corpus(store, CORPUS, gold, variant="synthetic", channels=channels)
    m = arb.corpus_metrics(block)
    by = {s.strategy: s for s in block.strategies}
    assert m["S3_accuracy"] == pytest.approx(by["S3"].accuracy)
    assert m["oracle_gap_S3_minus_S2"] == pytest.approx(by["S3"].accuracy - by["S2"].accuracy)
    text = next(c for c in block.channels if c.modality == "text")
    assert m["channel_text_accuracy"] == pytest.approx(text.accuracy)
    comp = next(c for c in block.complementarity
                if (c.modality_a, c.modality_b) == ("text", "speech"))
    assert m["kappa_text_speech"] == pytest.approx(comp.kappa_predictions)
    assert m["pair_oracle_text_speech"] == pytest.approx(comp.pair_oracle_accuracy)
