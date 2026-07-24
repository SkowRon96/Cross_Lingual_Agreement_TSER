"""Per-emotion correct/incorrect breakdown for every (study, corpus, model).

For each gold emotion: n_gold, correct (pred==gold), incorrect (pred!=gold, incl. abstain),
and pct_correct (recall = correct / n_gold). Writes one tidy master CSV plus a readable
markdown table per (study, corpus, model). Reads only cached predictions + the ingested
index gold — recompute any time with:  .venv/bin/python scripts/per_emotion_breakdown.py
"""
import csv
from pathlib import Path

from app.analysis.metrics import baseline
from app.analysis.run import gold_lookup
from app.config import studies
from app.config.emotions import CANONICAL_ORDER
from app.config.settings import Settings
from app.predict.store import PredictionStore

ORDER = [e.value for e in CANONICAL_ORDER]


def rows_for(confusion: dict[str, dict[str, int]]):
    """confusion[gold][pred] -> list of (emotion, n_gold, correct, incorrect, pct)."""
    out = []
    tot_n = tot_c = 0
    for emo in ORDER:
        row = confusion.get(emo)
        if not row:
            continue
        n_gold = sum(row.values())
        correct = row.get(emo, 0)
        incorrect = n_gold - correct
        pct = 100.0 * correct / n_gold if n_gold else 0.0
        out.append((emo, n_gold, correct, incorrect, pct))
        tot_n += n_gold
        tot_c += correct
    out.append(("TOTAL", tot_n, tot_c, tot_n - tot_c,
                100.0 * tot_c / tot_n if tot_n else 0.0))
    return out


def main():
    s = Settings()
    store = PredictionStore(s.results_root / "predictions")
    out_dir = s.results_root / "tables" / "per_emotion"
    out_dir.mkdir(parents=True, exist_ok=True)
    master = s.results_root / "tables" / "per_emotion_breakdown.csv"

    tidy = [("study", "corpus", "model", "emotion", "n_gold", "correct", "incorrect", "pct_correct")]
    n_tables = 0
    for (study, corpus), model_ids in sorted(studies.ROSTER.items()):
        try:
            gold = gold_lookup(study, corpus, s.index_dir)
        except FileNotFoundError:
            continue
        for mid in model_ids:
            if not store.path(study, corpus, mid).exists():
                continue
            preds = store.read(study, corpus, mid)
            b = baseline(preds, gold, mid, corpus, studies.MODALITY[mid])
            rows = rows_for(b.confusion)
            for emo, n_gold, correct, incorrect, pct in rows:
                tidy.append((study, corpus, mid, emo, n_gold, correct, incorrect, f"{pct:.1f}"))
            # readable markdown per model
            md = [f"# {mid} on {corpus} (study {study}) — n={b.n}, acc={b.accuracy:.3f}", "",
                  "| emotion | n_gold | correct | incorrect | % correct |",
                  "|:--------|-------:|--------:|----------:|----------:|"]
            for emo, n_gold, correct, incorrect, pct in rows:
                md.append(f"| {emo} | {n_gold} | {correct} | {incorrect} | {pct:.1f}% |")
            (out_dir / f"study{study}_{corpus}_{mid}.md").write_text("\n".join(md) + "\n")
            n_tables += 1

    with master.open("w", newline="") as f:
        csv.writer(f).writerows(tidy)
    print(f"wrote {n_tables} per-model tables -> {out_dir}")
    print(f"master tidy CSV -> {master} ({len(tidy) - 1} rows)")


if __name__ == "__main__":
    main()
