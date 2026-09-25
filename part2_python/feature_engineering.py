from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def configure_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure console and file logging for this script entry point."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

LOW_VISIBILITY_WEATHER = {"Fog", "Mist", "Haze", "Smoke", "Squall"}
SEVERE_WEATHER = {"Thunderstorm", "Snow", "Squall", "Fog"}


def min_max_scale(series: pd.Series) -> pd.Series:
    """Scale a numeric series to [0, 1] using NumPy/Pandas."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - minimum) / (maximum - minimum)


def validate_input(df: pd.DataFrame) -> None:
    """Validate that the cleaned input has the expected columns."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Feature-engineering input schema validation passed")


def load_cleaned_data(input_path: Path) -> pd.DataFrame:
    """Load the cleaned dataset without converting the literal holiday 'None' to NaN."""
    try:
        df = pd.read_csv(input_path, keep_default_na=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cleaned dataset not found: {input_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parsing failed: {input_path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read cleaned dataset: {input_path}") from exc

    validate_input(df)
    logger.info("Loaded cleaned dataset: %d rows x %d columns", *df.shape)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-ready time, weather, scaling and congestion features."""
    logger.info("Shape before feature engineering: %s", df.shape)
    result = df.copy()

    result["date_time"] = pd.to_datetime(result["date_time"], errors="coerce")
    invalid_dates = int(result["date_time"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Feature engineering found {invalid_dates} invalid date_time values")

    # Time features.
    result["hour"] = result["date_time"].dt.hour.astype("int64")
    result["day_of_week"] = result["date_time"].dt.dayofweek.astype("int64")
    result["day_of_week_name"] = result["date_time"].dt.day_name()
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int64")
    result["month"] = result["date_time"].dt.month.astype("int64")
    result["is_holiday"] = (result["holiday"].astype(str).str.strip().str.lower() != "none").astype("int64")

    # Cyclical encodings for hour and day of week.
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24.0)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7.0)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7.0)

    # Weather features.
    weather = result["weather_main"].astype(str).str.strip()
    result["is_low_visibility"] = weather.isin(LOW_VISIBILITY_WEATHER).astype("int64")
    result["is_severe_weather"] = weather.isin(SEVERE_WEATHER).astype("int64")
    result["has_precipitation"] = (
        (pd.to_numeric(result["rain_1h"], errors="coerce") > 0)
        | (pd.to_numeric(result["snow_1h"], errors="coerce") > 0)
    ).astype("int64")

    weather_dummies = pd.get_dummies(
        weather,
        prefix="weather",
        dtype="int64",
    )
    result = pd.concat([result, weather_dummies], axis=1)
    logger.info("Encoded weather_main into %d one-hot features", weather_dummies.shape[1])

    # Min-max scaled continuous features.
    result["temp_scaled"] = min_max_scale(result["temp"])
    result["traffic_volume_scaled"] = min_max_scale(result["traffic_volume"])

    logger.debug(
        "Scaling ranges: temp=(%.3f, %.3f), traffic_volume=(%.3f, %.3f)",
        float(pd.to_numeric(result["temp"], errors="coerce").min()),
        float(pd.to_numeric(result["temp"], errors="coerce").max()),
        float(pd.to_numeric(result["traffic_volume"], errors="coerce").min()),
        float(pd.to_numeric(result["traffic_volume"], errors="coerce").max()),
    )

    # Data-driven congestion category using traffic-volume quartiles.
    traffic = pd.to_numeric(result["traffic_volume"], errors="coerce")
    q1, q2, q3 = traffic.quantile([0.25, 0.50, 0.75]).tolist()
    logger.debug(
        "Congestion quartile thresholds: Q1=%.3f, Q2=%.3f, Q3=%.3f",
        q1,
        q2,
        q3,
    )

    def bucket(value: float) -> str:
        if value <= q1:
            return "Low"
        if value <= q2:
            return "Medium"
        if value <= q3:
            return "High"
        return "Severe"

    result["congestion_category"] = traffic.apply(bucket)
    logger.info(
        "Created data-driven congestion_category using traffic-volume quartiles"
    )
    logger.debug(
        "Congestion category counts: %s",
        result["congestion_category"].value_counts().to_dict(),
    )

    logger.info("Shape after feature engineering: %s", result.shape)
    return result


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save engineered features to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Feature-engineered dataset saved: %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Engineer ML-ready traffic features.")
    parser.add_argument("--input", type=Path, required=True, help="Path to cleaned CSV")
    parser.add_argument("--output", type=Path, required=True, help="Path for engineered CSV")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path (default: part2_python/logs/pipeline.log)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for console and file output",
    )
    args = parser.parse_args()

    log_path = args.log
    if log_path is None:
        log_path = Path(__file__).resolve().parent / "logs" / "pipeline.log"
    configure_logging(log_path, getattr(logging, args.log_level))
    logger.info("Feature engineering started")

    try:
        df = load_cleaned_data(args.input)
        engineered = engineer_features(df)
        save_features(engineered, args.output)
        logger.info("Feature engineering completed successfully")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Feature engineering failed: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # graceful safety net for the CLI entry point
        logger.error("Unexpected feature-engineering failure: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
