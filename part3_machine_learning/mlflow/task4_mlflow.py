#!/usr/bin/env python3
"""Task 4 - Advanced AI Technique: MLflow experiment tracking.

This script tracks the already-trained Part 3 Task 1 models without retraining them.
It logs parameters, evaluation metrics, model artifacts, dataset metadata, and a
simple model-version tag to a local MLflow tracking store.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    import mlflow
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "MLflow is not installed. Run: pip install -q mlflow"
    ) from exc

LOGGER = logging.getLogger(__name__)


def configure_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track Task 1 models with MLflow")
    parser.add_argument("--input", required=True, help="Path to traffic_cleaned.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--metrics", default="results/task1_model_metrics.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--tracking-dir", default="mlruns")
    parser.add_argument("--log", default="logs/task4.log")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
        return numeric if pd.notna(numeric) else None
    except (TypeError, ValueError):
        return None


def model_metadata(model_name: str) -> dict[str, Any]:
    """Extract a small, JSON-friendly set of hyperparameters for tracking."""
    path = Path(model_name)
    return {"model_file": str(path)}


def track_models(
    input_path: Path,
    models_dir: Path,
    metrics_path: Path,
    output_dir: Path,
    tracking_dir: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Task 1 metrics file not found: {metrics_path}")

    df = pd.read_csv(input_path)
    metrics = pd.read_csv(metrics_path)
    LOGGER.info("Loaded Task 4 dataset: %d rows x %d columns", *df.shape)
    LOGGER.info("Loaded Task 1 metrics: %d model records", len(metrics))

    tracking_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_db = tracking_dir.parent / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_db}"
    mlflow.set_tracking_uri(tracking_uri)
    LOGGER.info("MLflow tracking URI: %s", tracking_uri)
    experiment_name = "Smart City Traffic Intelligence - Part 3"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    LOGGER.info("MLflow experiment ready: %s (id=%s)", experiment_name, experiment.experiment_id if experiment else "unknown")

    model_files = {
        "logistic_regression": models_dir / "logistic_regression.joblib",
        "random_forest_classifier": models_dir / "random_forest_classifier.joblib",
        "linear_regression": models_dir / "linear_regression.joblib",
        "hist_gradient_boosting_regressor": models_dir / "hist_gradient_boosting_regressor.joblib",
    }

    records: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        model_name = str(row["model"])
        if model_name not in model_files:
            LOGGER.warning("Skipping unrecognised model in metrics: %s", model_name)
            continue

        model_path = model_files[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found: {model_path}")

        task = str(row["task"])
        version_tag = f"task1-v1-{model_name}"
        model = joblib.load(model_path)

        with mlflow.start_run(run_name=f"{task}-{model_name}") as run:
            params = {
                "task": task,
                "model_name": model_name,
                "model_version": version_tag,
                "dataset_rows": len(df),
                "dataset_columns": df.shape[1],
                "random_state": 42,
            }
            if hasattr(model, "n_estimators"):
                params["n_estimators"] = int(model.n_estimators)
            if hasattr(model, "max_depth") and getattr(model, "max_depth") is not None:
                params["max_depth"] = int(model.max_depth)
            if hasattr(model, "hidden_layer_sizes"):
                params["hidden_layer_sizes"] = str(model.hidden_layer_sizes)
            if hasattr(model, "max_iter"):
                params["max_iter"] = int(model.max_iter)

            mlflow.log_params(params)
            mlflow.set_tags({
                "project": "smart-city-traffic-intelligence",
                "part": "3",
                "advanced_technique": "MLflow experiment tracking",
                "model_version": version_tag,
                "target_note": "Classification uses the documented proxy high_risk label; not actual accident occurrence.",
            })

            metric_names = (
                ["accuracy", "precision", "recall", "f1", "roc_auc"]
                if task == "classification"
                else ["mae", "r2"]
            )
            logged: dict[str, float] = {}
            for metric_name in metric_names:
                value = safe_float(row.get(metric_name))
                if value is not None:
                    mlflow.log_metric(metric_name, value)
                    logged[metric_name] = value

            # Log the already-trained .joblib file as an artifact instead of using
            # mlflow.sklearn.log_model(). Recent MLflow/skops versions may reject
            # otherwise safe scikit-learn objects as untrusted types during model
            # serialization. The original .joblib artifact plus the model_version
            # tag provides a reproducible tracked model without that dependency.
            mlflow.log_artifact(str(model_path), artifact_path="source_model")
            LOGGER.debug("Logged source model artifact: %s", model_path)

            summary = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "task": task,
                "model": model_name,
                "model_version": version_tag,
                "metrics": logged,
            }
            records.append(summary)
            LOGGER.info(
                "MLflow run saved: model=%s task=%s run_id=%s metrics=%s",
                model_name, task, run.info.run_id, logged
            )

    summary_df = pd.DataFrame(records)
    summary_path = output_dir / "task4_mlflow_runs.csv"
    summary_df.to_csv(summary_path, index=False)
    (output_dir / "task4_mlflow_summary.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    LOGGER.info("MLflow summary saved: %s", summary_path)
    LOGGER.info("Tracked %d models/runs in MLflow", len(records))
    return summary_df


def main() -> int:
    args = parse_args()
    configure_logging(Path(args.log), args.log_level)
    LOGGER.info("Part 3 Task 4 MLflow tracking started")
    try:
        track_models(
            input_path=Path(args.input),
            models_dir=Path(args.models_dir),
            metrics_path=Path(args.metrics),
            output_dir=Path(args.output_dir),
            tracking_dir=Path(args.tracking_dir),
        )
        LOGGER.info("Part 3 Task 4 MLflow tracking completed successfully")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Task 4 MLflow tracking failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
