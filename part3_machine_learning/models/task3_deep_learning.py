#!/usr/bin/env python3
"""Part 3 Task 3: Neural-network demand prediction with SHAP explainability."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def configure_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    required = ["date_time", "holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "traffic_volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["date_time"] = pd.to_datetime(data["date_time"], errors="coerce")
    if data["date_time"].isna().any():
        raise ValueError("date_time contains invalid or missing values")

    data["hour"] = data["date_time"].dt.hour
    data["day_of_week"] = data["date_time"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_holiday"] = data["holiday"].notna().astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["day_of_week_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["day_of_week_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

    numeric = [
        "temp", "rain_1h", "snow_1h", "clouds_all",
        "hour", "day_of_week", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    ]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    weather_dummies = pd.get_dummies(data["weather_main"].fillna("Unknown"), prefix="weather", dtype=int)
    X = pd.concat([data[numeric], weather_dummies], axis=1)
    y = pd.to_numeric(data["traffic_volume"], errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()
    dropped = int((~valid).sum())
    if dropped:
        logger.warning("Dropped %d rows with missing feature/target values", dropped)
        X = X.loc[valid]
        y = y.loc[valid]

    logger.info("Built Task 3 feature set: %d rows x %d features", X.shape[0], X.shape[1])
    logger.info("Feature set includes time, weather encodings, holiday flag, and cyclical hour/day features")
    return X, y


def run(input_path: Path, output_dir: Path, log_path: Path, log_level: str) -> None:
    configure_logging(log_path, log_level)
    try:
        logger.info("Part 3 Task 3 deep learning started")
        df = pd.read_csv(input_path)
        logger.info("Loaded input dataset: %d rows x %d columns", df.shape[0], df.shape[1])

        X, y = build_features(df)
        if len(X) > 12000:
            sample_idx = X.sample(n=12000, random_state=42).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]
            logger.info("Sampled 12000 observations for efficient neural-network training")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Train/test split created: train=%d, test=%d", len(X_train), len(X_test))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Feature scaling completed for neural network")

        nn = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="lbfgs",
            alpha=0.0001,
            max_iter=300,
            random_state=42,
        )
        nn.fit(X_train_scaled, y_train)
        pred = nn.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        logger.info("Neural network trained: iterations=%d, MAE=%.4f, R2=%.4f", nn.n_iter_, mae, r2)

        models_dir = output_dir / "models"
        results_dir = output_dir / "results"
        figures_dir = output_dir / "figures"
        for d in (models_dir, results_dir, figures_dir):
            d.mkdir(parents=True, exist_ok=True)

        joblib.dump(nn, models_dir / "task3_neural_network_mlp.joblib")
        joblib.dump(scaler, models_dir / "task3_neural_network_scaler.joblib")
        logger.info("Saved neural-network model and scaler")

        metrics = pd.DataFrame([{
            "model": "MLPRegressor_neural_network",
            "mae": mae,
            "r2": r2,
            "iterations": nn.n_iter_,
            "hidden_layers": "32,16",
            "random_state": 42,
        }])
        metrics.to_csv(results_dir / "task3_neural_network_metrics.csv", index=False)
        logger.info("Saved neural-network metrics: %s", results_dir / "task3_neural_network_metrics.csv")

        # Explainability: comparable tree model on the same target and feature set.
        logger.info("Training comparable Random Forest regressor for SHAP explainability")
        rf = RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1, max_depth=12, min_samples_leaf=2)
        rf.fit(X_train, y_train)
        logger.info("Comparable Random Forest trained for SHAP")
        joblib.dump(rf, models_dir / "task3_shap_reference_random_forest.joblib")

        sample_n = min(300, len(X_test))
        X_shap = X_test.sample(n=sample_n, random_state=42)
        logger.info("Computing SHAP values using %d test observations", sample_n)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_shap)

        shap_array = np.asarray(shap_values)
        mean_abs = np.mean(np.abs(shap_array), axis=0)
        importance = pd.DataFrame({
            "feature": X_shap.columns,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(results_dir / "task3_shap_feature_importance.csv", index=False)
        logger.info("Saved SHAP feature importance table")

        plt.figure(figsize=(9, 6))
        top = importance.head(12).sort_values("mean_abs_shap")
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Mean absolute SHAP value")
        plt.title("Task 3: SHAP Feature Importance (Comparable Tree Model)")
        plt.tight_layout()
        shap_fig = figures_dir / "task3_shap_feature_importance.png"
        plt.savefig(shap_fig, dpi=180)
        plt.close()
        logger.info("Figure saved: %s", shap_fig)

        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, pred, alpha=0.25, s=10)
        lo = float(min(y_test.min(), pred.min()))
        hi = float(max(y_test.max(), pred.max()))
        plt.plot([lo, hi], [lo, hi], linewidth=1.5)
        plt.xlabel("Observed traffic volume")
        plt.ylabel("Predicted traffic volume")
        plt.title(f"Task 3: Neural Network Demand Prediction (R2={r2:.3f}, MAE={mae:.1f})")
        plt.tight_layout()
        pred_fig = figures_dir / "task3_neural_network_predictions.png"
        plt.savefig(pred_fig, dpi=180)
        plt.close()
        logger.info("Figure saved: %s", pred_fig)

        logger.info("Explainability note: SHAP is applied to a comparable Random Forest trained on the same features and target because tree-based SHAP is more direct to interpret than the MLP")
        logger.info("Part 3 Task 3 completed successfully")
    except Exception as exc:
        logger.error("Task 3 failed: %s", exc, exc_info=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output_dir), Path(args.log), args.log_level)


if __name__ == "__main__":
    main()
