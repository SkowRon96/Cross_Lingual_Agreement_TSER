"""Command-line entry point for the research pipeline."""
import argparse
from pathlib import Path

from app.config.settings import Settings
from app.data.index_io import read_index
from app.models.registry import build_model
from app.predict.runner import run_prediction
from app.predict.store import PredictionStore


def _predict(args) -> int:
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    index = Path(args.index) if args.index else settings.index_dir / f"{args.corpus}.parquet"
    samples = read_index(index)
    model = build_model(args.model)
    store = PredictionStore(results_root / "predictions")
    status = run_prediction(samples, model, store, args.study, args.corpus,
                            force=args.force, batch_size=args.batch_size)
    print(f"predict {args.model} on {args.corpus} (study {args.study}): "
          f"{status}, {len(samples)} samples -> {store.path(args.study, args.corpus, args.model)}")
    return 0


def _import_predictions(args) -> int:
    from app.data.import_preds import import_predictions
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    store = PredictionStore(results_root / "predictions")
    n = import_predictions(Path(args.file), store, args.study, args.corpus)
    print(f"imported {n} speech_zeroshot predictions for {args.corpus} (study {args.study})")
    return 0


def _analyze(args) -> int:
    from app.analysis.run import analyze_study
    from app.analysis.results import write_json, write_tidy_csv
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    index_dir = Path(args.index_dir) if args.index_dir else settings.index_dir
    bundle = analyze_study(args.study, results_root=results_root, index_dir=index_dir)
    out = results_root / "analysis" / f"study{args.study}"
    write_json(bundle, out / "bundle.json")
    write_tidy_csv(bundle, out / "agreements.csv")
    print(f"analyze study {args.study}: {len(bundle.agreements)} agreements, "
          f"{len(bundle.baselines)} baselines -> {out}")
    return 0


def _report(args) -> int:
    from app.report.run import report_study
    settings = Settings()
    results_root = Path(args.results_root) if args.results_root else settings.results_root
    paths = report_study(args.study, results_root=results_root)
    print(f"report study {args.study}: wrote {len(paths)} artefacts -> "
          f"{results_root / 'tables'} + {results_root / 'figures'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Run a model over a corpus index")
    p.add_argument("--study", type=int, required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--index", default=None)
    p.add_argument("--results-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=_predict)
    ip = sub.add_parser("import-predictions", help="Import external predictions (e.g. Qwen from Windows) into the store")
    ip.add_argument("--file", required=True)
    ip.add_argument("--study", type=int, required=True)
    ip.add_argument("--corpus", required=True)
    ip.add_argument("--results-root", default=None)
    ip.set_defaults(func=_import_predictions)
    an = sub.add_parser("analyze", help="Compute analysis (kappa, tests, conflicts) into results/analysis")
    an.add_argument("--study", type=int, required=True)
    an.add_argument("--results-root", default=None)
    an.add_argument("--index-dir", default=None)
    an.set_defaults(func=_analyze)
    rp = sub.add_parser("report", help="Render figures + tables from results/analysis")
    rp.add_argument("--study", type=int, required=True)
    rp.add_argument("--results-root", default=None)
    rp.set_defaults(func=_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
