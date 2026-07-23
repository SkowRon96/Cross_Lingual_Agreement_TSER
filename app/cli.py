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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
