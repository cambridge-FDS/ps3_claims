"""Run the PS3 benchmark or the PS4 Ex1 monotonicity analysis."""

import argparse
import json
import hashlib
import shutil
import socket
import subprocess
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from analyses.model_comparison import save_comparison, write_metrics_csv
from analyses.part_c import (
    prepare_model_data,
    run_baseline,
    run_initial_lightgbm,
    run_tuned_lightgbm,
    sample_frames,
    write_model_metrics,
)
def load_model_data(frame=None):
    """Apply the canonical target and Member A's agreed policy split."""
    return prepare_model_data(frame)


def save_run(metrics, test_predictions, results_dir):
    """Save the audit metrics and a gitignored test-predictions file."""
    path = write_model_metrics(metrics, results_dir / "evaluated")
    test_predictions.rename("PurePremiumPredicted").rename_axis("IDpol").to_csv(
        results_dir / f"{metrics['model']}_predictions.csv"
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved {path}")


def run_provenance(frame):
    """Fingerprint the exact prepared values and executed modelling sources."""
    ordered = frame.sort_values("IDpol")
    digest = hashlib.sha256()
    digest.update(str(list(zip(ordered.columns, ordered.dtypes.astype(str)))).encode())
    digest.update(pd.util.hash_pandas_object(ordered, index=False).values.tobytes())
    sources = [*Path("analyses").glob("*.py"), *Path("ps3").rglob("*.py")]
    return digest.hexdigest(), {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        "versions": {name: version(name) for name in ["glum", "dask-ml", "lightgbm", "numpy", "pandas", "scikit-learn"]},
    }


def archive_results(results_dir):
    """Keep prior review evidence locally before a command replaces outputs."""
    if not results_dir.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path("data/run_backups") / stamp
    shutil.copytree(results_dir, destination)
    print(f"Archived prior outputs to {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="benchmark",
        choices=["check-data", "benchmark", "lightgbm", "tuned-lightgbm", "parametric", "metrics", "compare", "all", "ps4-ex1"],
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, help="Explicit local prepared Parquet cache; otherwise use the instructor loader")
    parser.add_argument("--save-data", type=Path, help="Save the shared prepared frame locally for independent rescoring")
    parser.add_argument("--network-timeout", type=float, default=300., help="Remote socket read timeout in seconds")
    args = parser.parse_args(argv)

    if args.action == "metrics":
        archive_results(args.results_dir)
        print(f"Saved {write_metrics_csv(args.results_dir)}")
        return

    if args.network_timeout <= 0:
        parser.error("--network-timeout must be positive")
    previous_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(args.network_timeout)
        frame = load_model_data(pd.read_parquet(args.data) if args.data else None)
    finally:
        socket.setdefaulttimeout(previous_timeout)
    if args.save_data:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        if args.save_data.exists():
            raise FileExistsError("Choose a new cache path to preserve the existing data cache")
        frame.to_parquet(args.save_data, index=False)
    if args.action == "check-data":
        train, test = sample_frames(frame)
        print(f"Validated {len(frame):,} policies ({len(train):,} train; {len(test):,} test)")
        return
    # ==================== PS4 begins: Exercise 1 ====================
    if args.action == "ps4-ex1":
        from analyses.lgbm_monotonicity import run_monotonicity_analysis

        _, _, metrics, _ = run_monotonicity_analysis(frame, args.results_dir)
        print(json.dumps(metrics, indent=2, allow_nan=False))
        return
    # ===================== PS4 ends: Exercise 1 =====================
    archive_results(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_fingerprint, provenance = run_provenance(frame)
    def save_verified(metrics, predictions):
        metrics.update(data_fingerprint=data_fingerprint, run_provenance=provenance,
                       verification_status="locally_trained")
        save_run(metrics, predictions, args.results_dir)

    if args.action in {"benchmark", "all"}:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            metrics, coefficients, _, predictions = run_baseline(frame)
        print(coefficients.to_string())
        coefficients.rename_axis("feature").to_csv(args.results_dir / "tweedie_glm_coefficients.csv")
        save_verified(metrics, predictions)
    if args.action in {"lightgbm", "all"}:
        metrics, predictions = run_initial_lightgbm(frame)
        save_verified(metrics, predictions)
    if args.action in {"parametric", "all"}:
        from analyses.parametric_glm import run_parametric_glm
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            metrics, _, predictions = run_parametric_glm(frame)
        metrics["evaluation_caveat"] = "D's specification was chosen after test-tail diagnostics; this test score is exploratory, not untouched validation."
        save_verified(metrics, predictions)
    if args.action in {"tuned-lightgbm", "all"}:
        audit_dir = Path("data/tuning_runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metrics, predictions = run_tuned_lightgbm(frame, audit_dir)
        save_verified(metrics, predictions)
    if args.action in {"compare", "all"}:
        train, test = sample_frames(frame)
        from analyses.model_comparison import read_model_metrics
        existing = read_model_metrics(args.results_dir)
        if "data_fingerprint" in existing and any(existing["data_fingerprint"].dropna() != data_fingerprint):
            raise ValueError("Saved metrics do not match this prepared dataset")
        save_comparison(train, test, args.results_dir)


if __name__ == "__main__":
    main()
