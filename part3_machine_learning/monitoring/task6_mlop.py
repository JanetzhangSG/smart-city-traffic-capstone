from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def configure_logging(log_path: Path, level: str = "INFO") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(fmt)
        handler.setLevel(getattr(logging, level))
        LOGGER.addHandler(handler)


def load_cleaned(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input dataset not found: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parsing failed: {path}") from exc
    if "date_time" not in df.columns or "traffic_volume" not in df.columns:
        raise ValueError("Input dataset must contain date_time and traffic_volume")
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    if df["date_time"].isna().any():
        raise ValueError("Invalid date_time values found")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    if df["traffic_volume"].isna().any():
        raise ValueError("Non-numeric traffic_volume values found")
    df = df.sort_values("date_time").reset_index(drop=True)
    LOGGER.info("Loaded monitoring dataset: %d rows x %d columns", *df.shape)
    return df


def version_registry(metrics_path: Path, models_dir: Path, out_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    rows: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        model_name = str(row["model"])
        model_path = models_dir / f"{model_name}.joblib"
        if not model_path.exists():
            LOGGER.warning("Model artifact missing for %s: %s", model_name, model_path)
            continue
        performance = {
            "accuracy": row.get("accuracy"),
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "f1": row.get("f1"),
            "roc_auc": row.get("roc_auc"),
            "mae": row.get("mae"),
            "r2": row.get("r2"),
        }
        rows.append(
            {
                "model_version": f"task1-v1-{model_name}",
                "model": model_name,
                "task": row["task"],
                "status": "validated",
                "artifact": str(model_path),
                **performance,
            }
        )
    registry = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out_path, index=False)
    LOGGER.info("Saved model version registry: %s (%d records)", out_path, len(registry))
    return registry


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().astype(float)
    if ref.empty or cur.empty:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        lo, hi = float(ref.min()), float(ref.max())
        if math.isclose(lo, hi):
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype(str).fillna("<missing>")
    cur = current.astype(str).fillna("<missing>")
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    if not cats:
        return 0.0
    ref_counts = ref.value_counts(normalize=True).reindex(cats, fill_value=0).astype(float)
    cur_counts = cur.value_counts(normalize=True).reindex(cats, fill_value=0).astype(float)
    ref_pct = np.clip(ref_counts.values, 1e-6, None)
    cur_pct = np.clip(cur_counts.values, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def monitor_model(
    df: pd.DataFrame,
    model_path: Path,
    out_json: Path,
    baseline_fraction: float = 0.7,
    alert_psi: float = 0.20,
) -> dict[str, object]:
    model = joblib.load(model_path)
    split = max(1, min(len(df) - 1, int(len(df) * baseline_fraction)))
    baseline = df.iloc[:split].copy()
    current = df.iloc[split:].copy()

    feature_candidates_numeric = ["hour", "day_of_week", "month", "temp", "rain_1h", "snow_1h", "clouds_all"]
    work = df.copy()
    work["hour"] = work["date_time"].dt.hour
    work["day_of_week"] = work["date_time"].dt.dayofweek
    work["month"] = work["date_time"].dt.month
    baseline_work = work.iloc[:split]
    current_work = work.iloc[split:]

    psi_values: dict[str, float] = {}
    for col in feature_candidates_numeric:
        psi_values[col] = psi_numeric(baseline_work[col], current_work[col])
    for col in ["weather_main", "weather_description", "holiday"]:
        if col in work.columns:
            psi_values[col] = psi_categorical(baseline_work[col], current_work[col])

    def make_model_features(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.copy()
        x["hour"] = x["date_time"].dt.hour
        x["day_of_week"] = x["date_time"].dt.dayofweek
        x["is_weekend"] = (x["day_of_week"] >= 5).astype(int)
        x["month"] = x["date_time"].dt.month
        x["is_holiday"] = (x["holiday"].astype(str).str.strip().str.lower() != "none").astype(int)
        x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24.0)
        x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24.0)
        x["day_sin"] = np.sin(2 * np.pi * x["day_of_week"] / 7.0)
        x["day_cos"] = np.cos(2 * np.pi * x["day_of_week"] / 7.0)
        weather = x["weather_main"].astype(str).str.strip()
        low_visibility = {"Fog", "Mist", "Haze", "Smoke", "Squall"}
        severe_weather = {"Thunderstorm", "Snow", "Squall", "Fog"}
        x["is_low_visibility"] = weather.isin(low_visibility).astype(int)
        x["is_severe_weather"] = weather.isin(severe_weather).astype(int)
        x["has_precipitation"] = (
            (pd.to_numeric(x["rain_1h"], errors="coerce") > 0)
            | (pd.to_numeric(x["snow_1h"], errors="coerce") > 0)
        ).astype(int)
        return x[[
            "hour", "day_of_week", "is_weekend", "month", "is_holiday",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "weather_main", "weather_description", "rain_1h", "snow_1h",
            "clouds_all", "is_low_visibility", "is_severe_weather",
            "has_precipitation", "temp",
        ]]

    X_baseline = make_model_features(baseline)
    X_current = make_model_features(current)
    baseline_pred = model.predict(X_baseline)
    current_pred = model.predict(X_current)
    baseline_mae = float(np.mean(np.abs(baseline["traffic_volume"].to_numpy() - baseline_pred)))
    current_mae = float(np.mean(np.abs(current["traffic_volume"].to_numpy() - current_pred)))
    mae_ratio = float(current_mae / baseline_mae) if baseline_mae > 0 else float("inf")
    max_psi_feature = max(psi_values, key=psi_values.get)
    max_psi = float(psi_values[max_psi_feature])

    alerts: list[str] = []
    if max_psi >= alert_psi:
        alerts.append(f"Feature drift: {max_psi_feature} PSI={max_psi:.3f}")
    if mae_ratio >= 1.50:
        alerts.append(f"Prediction error drift: MAE ratio={mae_ratio:.3f}")

    status = "ALERT / Requires investigation" if alerts else "PASS / Normal"
    report = {
        "model_version": "task1-v1-hist_gradient_boosting_regressor",
        "monitoring_method": "chronological baseline/current comparison",
        "baseline_rows": int(len(baseline)),
        "current_rows": int(len(current)),
        "baseline_mae": baseline_mae,
        "current_mae": current_mae,
        "mae_ratio": mae_ratio,
        "psi_threshold_alert": alert_psi,
        "feature_psi": psi_values,
        "max_psi_feature": max_psi_feature,
        "max_psi": max_psi,
        "alert_if_mae_ratio_ge": 1.50,
        "status": status,
        "alerts": alerts,
        "note": "Monitoring is a capstone simulation using chronological slices of the supplied corridor dataset; thresholds are illustrative and not production policy."
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    if alerts:
        LOGGER.warning("Monitoring alert triggered: %s", "; ".join(alerts))
    else:
        LOGGER.info("Monitoring status PASS / Normal")
    LOGGER.info("Saved monitoring report: %s", out_json)
    return report


def log_deployment_experiment(
    report: dict[str, object],
    registry_path: Path,
    tracking_db: Path,
    output_dir: Path,
) -> str:
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    experiment_name = "Smart City Traffic Intelligence - Part 3"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="task6-deployment-monitoring") as run:
        mlflow.set_tag("stage", "deployment_simulation")
        mlflow.set_tag("model_version", str(report["model_version"]))
        mlflow.log_param("monitoring_method", str(report["monitoring_method"]))
        mlflow.log_param("psi_alert_threshold", float(report["psi_threshold_alert"]))
        mlflow.log_param("mae_ratio_alert_threshold", float(report["alert_if_mae_ratio_ge"]))
        mlflow.log_metric("baseline_mae", float(report["baseline_mae"]))
        mlflow.log_metric("current_mae", float(report["current_mae"]))
        mlflow.log_metric("mae_ratio", float(report["mae_ratio"]))
        mlflow.log_metric("max_psi", float(report["max_psi"]))
        mlflow.set_tag("alert_status", str(report["status"]))
        mlflow.log_artifact(str(registry_path), artifact_path="model_registry")
        mlflow.log_artifact(str(output_dir / "task6_monitoring_report.json"), artifact_path="monitoring")
        LOGGER.info("MLflow deployment/monitoring run saved: %s", run.info.run_id)
        return run.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 3 Task 6 MLOps, versioning and monitoring simulation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    configure_logging(args.log, args.log_level)
    LOGGER.info("Part 3 Task 6 MLOps simulation started")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        df = load_cleaned(args.input)
        registry = version_registry(args.metrics, args.models_dir, args.output_dir / "task6_model_version_registry.csv")
        reg_model = args.models_dir / "hist_gradient_boosting_regressor.joblib"
        if not reg_model.exists():
            raise FileNotFoundError(f"Deployment candidate model not found: {reg_model}")
        report = monitor_model(
            df,
            reg_model,
            args.output_dir / "task6_monitoring_report.json",
        )
        run_id = log_deployment_experiment(report, args.output_dir / "task6_model_version_registry.csv", args.tracking_db, args.output_dir)
        summary = {
            "deployment_model_version": report["model_version"],
            "mlflow_run_id": run_id,
            "monitoring_status": report["status"],
            "baseline_mae": report["baseline_mae"],
            "current_mae": report["current_mae"],
            "max_psi": report["max_psi"],
        }
        (args.output_dir / "task6_mlop_summary.json").write_text(json.dumps(summary, indent=2))
        if report["status"].startswith("ALERT"):
            LOGGER.warning("Task 6 completed with monitoring status ALERT / Requires investigation")
        else:
            LOGGER.info("Task 6 completed with monitoring status PASS / Normal")
        LOGGER.info("Task 6 MLOps simulation completed successfully")
        return 0
    except Exception as exc:
        LOGGER.error("Task 6 MLOps simulation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
