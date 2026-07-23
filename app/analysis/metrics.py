"""Per-model baseline metrics: accuracy, macro-F1, confusion matrix (canonical labels)."""
from collections import defaultdict

from app.analysis.results import Baseline
from app.config.emotions import Emotion
from app.models.base import Prediction


def baseline(preds: list[Prediction], gold: dict[str, Emotion],
             model_id: str, corpus: str, modality: str) -> Baseline:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    n = correct = 0
    for p in preds:
        if p.sample_id not in gold:
            continue
        g = gold[p.sample_id]
        n += 1
        pred_key = p.pred.value if p.pred is not None else "abstain"
        confusion[g.value][pred_key] += 1
        if p.pred is None:
            fn[g.value] += 1
            continue
        if p.pred == g:
            correct += 1
            tp[g.value] += 1
        else:
            fp[p.pred.value] += 1
            fn[g.value] += 1
    gold_classes = sorted({g.value for g in gold.values()})
    f1s = []
    for c in gold_classes:
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append((2 * tp[c] / denom) if denom else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return Baseline(
        model_id=model_id, corpus=corpus, modality=modality, n=n,
        accuracy=(correct / n if n else 0.0), macro_f1=macro_f1,
        confusion={g: dict(row) for g, row in confusion.items()},
        labels=gold_classes,
    )
