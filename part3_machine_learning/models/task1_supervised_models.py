from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

EXPECTED = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume"
]

# The brief references SEVERE_WEATHER but does not enumerate its values.
# This project uses the following operational definition and documents it in the report.
SEVERE_WEATHER = {"Thunderstorm", "Snow", "Squall", "Fog"}
LOW_VISIBILITY_WEATHER = {"Fog", "Mist", "Haze", "Smoke", "Squall"}


def configure_logging(log_path: Path, level: int = logging.INFO) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()
    for h in (fh, ch):
        h.setLevel(level)
        h.setFormatter(fmt)
        logger.addHandler(h)


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input dataset not found: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parsing failed: {path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read input dataset: {path}") from exc
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Loaded supervised-learning input: %d rows x %d columns", *df.shape)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x = df.copy()
    x["date_time"] = pd.to_datetime(x["date_time"], errors="coerce")
    if x["date_time"].isna().any():
        raise ValueError("Invalid date_time values found in supervised-learning input")

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
    x["is_low_visibility"] = weather.isin(LOW_VISIBILITY_WEATHER).astype(int)
    x["is_severe_weather"] = weather.isin(SEVERE_WEATHER).astype(int)
    x["has_precipitation"] = (
        (pd.to_numeric(x["rain_1h"], errors="coerce") > 0)
        | (pd.to_numeric(x["snow_1h"], errors="coerce") > 0)
    ).astype(int)

    # Proxy accident-risk label, following the brief's prescribed construction.
    q1, q2, q3 = x["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    def bucket(v: float) -> str:
        if v <= q1:
            return "Low"
        if v <= q2:
            return "Medium"
        if v <= q3:
            return "High"
        return "Severe"
    x["congestion_category"] = x["traffic_volume"].apply(bucket)
    high_congestion = x["congestion_category"].isin(["High", "Severe"])
    risky_weather = x["weather_main"].isin(SEVERE_WEATHER) | (x["is_low_visibility"] == 1)
    y_class = (high_congestion & risky_weather).astype(int)

    # Common feature set: time, weather, holiday and cyclical encodings.
    # Exclude traffic_volume, congestion_category and their direct derivatives from classification to avoid target leakage.
    feature_columns = [
        "hour", "day_of_week", "is_weekend", "month", "is_holiday",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "weather_main", "weather_description", "rain_1h", "snow_1h", "clouds_all",
        "is_low_visibility", "is_severe_weather", "has_precipitation",
        "temp",
    ]
    X = x[feature_columns].copy()
    y_reg = pd.to_numeric(x["traffic_volume"], errors="coerce")
    if y_reg.isna().any():
        raise ValueError("traffic_volume contains non-numeric values")

    logger.info("Proxy high-risk label created: %d positive / %d negative", int(y_class.sum()), int((y_class == 0).sum()))
    logger.debug("Congestion quartiles: Q1=%.3f, Q2=%.3f, Q3=%.3f", q1, q2, q3)
    logger.debug("SEVERE_WEATHER definition used: %s", sorted(SEVERE_WEATHER))
    return X, y_class, y_reg.to_frame(name="traffic_volume")


def make_preprocessor() -> ColumnTransformer:
    categorical = ["weather_main", "weather_description"]
    numeric = [
        "hour", "day_of_week", "is_weekend", "month", "is_holiday",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "rain_1h", "snow_1h",
        "clouds_all", "is_low_visibility", "is_severe_weather", "has_precipitation", "temp"
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ]
    )


def train_and_evaluate(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.DataFrame, out_dir: Path, seed: int = 42) -> pd.DataFrame:
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_class, y_reg["traffic_volume"], test_size=0.2, random_state=seed, stratify=y_class
    )
    logger.info("Train/test split created: train=%d, test=%d", len(X_train), len(X_test))

    clf_models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest_classifier": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced"),
    }
    reg_models = {
        "linear_regression": LinearRegression(),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=seed),
    }

    results = []
    for name, model in clf_models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        pipe.fit(X_train, yc_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        row = {
            "task": "classification",
            "model": name,
            "accuracy": accuracy_score(yc_test, pred),
            "precision": precision_score(yc_test, pred, zero_division=0),
            "recall": recall_score(yc_test, pred, zero_division=0),
            "f1": f1_score(yc_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(yc_test, prob),
            "mae": np.nan,
            "r2": np.nan,
        }
        results.append(row)
        joblib.dump(pipe, out_dir / "models" / f"{name}.joblib")
        logger.info("Saved classification model: %s", out_dir / "models" / f"{name}.joblib")

    for name, model in reg_models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        pipe.fit(X_train, yr_train)
        pred = pipe.predict(X_test)
        row = {
            "task": "regression",
            "model": name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "mae": mean_absolute_error(yr_test, pred),
            "r2": r2_score(yr_test, pred),
        }
        results.append(row)
        joblib.dump(pipe, out_dir / "models" / f"{name}.joblib")
        logger.info("Saved regression model: %s", out_dir / "models" / f"{name}.joblib")

    metrics = pd.DataFrame(results)
    metrics.to_csv(out_dir / "results" / "task1_model_metrics.csv", index=False)
    logger.info("Saved Task 1 metrics: %s", out_dir / "results" / "task1_model_metrics.csv")

    # Simple comparison plot for required reporting evidence.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    c = metrics[metrics.task == "classification"].set_index("model")["f1"]
    r = metrics[metrics.task == "regression"].set_index("model")["r2"]
    c.plot(kind="bar", ax=axes[0], title="Classification F1-score")
    r.plot(kind="bar", ax=axes[1], title="Regression R-squared")
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("R-squared")
    fig.tight_layout()
    fig_path = out_dir / "figures" / "task1_model_comparison.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", fig_path)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 3 Task 1 supervised ML models")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("models", "figures", "results"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)
    configure_logging(args.log, getattr(logging, args.log_level))
    logger.info("Part 3 Task 1 supervised modelling started")
    logger.info("Proxy-label note: no accident dataset was supplied; high_risk is a documented proxy only")
    try:
        df = load_data(args.input)
        X, y_class, y_reg = build_features(df)
        metrics = train_and_evaluate(X, y_class, y_reg, args.output_dir)
        logger.info("Part 3 Task 1 completed successfully")
        logger.info("Classification metrics summary: %s", metrics[metrics.task == "classification"].to_dict(orient="records"))
        logger.info("Regression metrics summary: %s", metrics[metrics.task == "regression"].to_dict(orient="records"))
        return 0
    except Exception as exc:
        logger.error("Part 3 Task 1 failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
