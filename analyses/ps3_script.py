"""Run the PS3 Exercise 3 benchmark and compare the models available locally."""

import argparse
import json
from pathlib import Path

from analyses.model_comparison import save_comparison, write_metrics_csv
from analyses.part_c import (
    SPLIT,
    prepare_data,
    run_baseline,
    run_initial_lightgbm,
    sample_frames,
    write_model_metrics,
)
from ps3.data import create_sample_column


def load_model_data(frame=None):
    """Apply the instructor's preparation and Member A's agreed policy split."""
    data = prepare_data(frame)
    return create_sample_column(
        data, id_columns=SPLIT["id_columns"], training_frac=SPLIT["training_fraction"]
    )


def save_run(metrics, test_predictions, results_dir):
    """Save the audit metrics and a gitignored test-predictions file."""
    path = write_model_metrics(metrics, results_dir)
    test_predictions.rename("PurePremiumPredicted").rename_axis("IDpol").to_csv(
        path.with_name(f"{metrics['model']}_predictions.csv")
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="benchmark",
        choices=["check-data", "benchmark", "lightgbm", "metrics", "compare", "all"],
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if args.action == "metrics":
        print(f"Saved {write_metrics_csv(args.results_dir)}")
        return

    frame = load_model_data()
    if args.action == "check-data":
        train, test = sample_frames(frame)
        print(f"Validated {len(frame):,} policies ({len(train):,} train; {len(test):,} test)")
        return
    if args.action in {"benchmark", "all"}:
        metrics, coefficients, _, predictions = run_baseline(frame)
        print(coefficients.to_string())
        save_run(metrics, predictions, args.results_dir)
    if args.action in {"lightgbm", "all"}:
        metrics, predictions = run_initial_lightgbm(frame)
        save_run(metrics, predictions, args.results_dir)
    if args.action in {"compare", "all"}:
        train, test = sample_frames(frame)
        save_comparison(train, test, args.results_dir)


if __name__ == "__main__":
    main()
