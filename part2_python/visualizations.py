"""Create and save Matplotlib visualisations for the traffic dataset.

The script reads the feature-engineered CSV and produces at least three
visualisations required by the capstone brief. Each saved figure is logged
with its output path. No print() is used for internal progress reporting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def configure_logging(log_path: Path, level: int) -> None:
    """Configure console and file logging for this entry-point script."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False


def load_features(input_path: Path) -> pd.DataFrame:
    """Load feature-engineered data and validate required columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {
        "date_time",
        "hour",
        "day_of_week_name",
        "is_weekend",
        "temp",
        "traffic_volume",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    if df["date_time"].isna().any():
        bad = int(df["date_time"].isna().sum())
        raise ValueError(f"Found {bad} invalid date_time values")

    logger.info("Loaded feature-engineered dataset: %d rows x %d columns", *df.shape)
    return df


def save_figure(fig: plt.Figure, path: Path, description: str) -> None:
    """Save a figure and log its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s (%s)", path, description)


def plot_traffic_by_hour(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean traffic volume by hour."""
    hourly = df.groupby("hour", as_index=False)["traffic_volume"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly["hour"], hourly["traffic_volume"])
    ax.set_title("Average Traffic Volume by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Traffic Volume (vehicles/hour)")
    ax.set_xticks(range(24))
    ax.grid(axis="y", alpha=0.25)

    path = output_dir / "traffic_by_hour.png"
    save_figure(fig, path, "hourly traffic demand")

    peak = hourly.loc[hourly["traffic_volume"].idxmax()]
    low = hourly.loc[hourly["traffic_volume"].idxmin()]
    logger.info(
        "Hourly interpretation: peak average traffic occurs around hour %d (%.2f), "
        "while the lowest occurs around hour %d (%.2f)",
        int(peak["hour"]),
        peak["traffic_volume"],
        int(low["hour"]),
        low["traffic_volume"],
    )


def plot_weekday_vs_weekend(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare weekday and weekend average traffic."""
    grouped = (
        df.assign(day_type=df["is_weekend"].map({0: "Weekday", 1: "Weekend"}))
        .groupby("day_type", as_index=False)["traffic_volume"]
        .mean()
    )
    grouped["sort_order"] = grouped["day_type"].map({"Weekday": 0, "Weekend": 1})
    grouped = grouped.sort_values("sort_order")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped["day_type"], grouped["traffic_volume"])
    ax.set_title("Average Traffic: Weekday vs Weekend")
    ax.set_xlabel("Day Type")
    ax.set_ylabel("Average Traffic Volume (vehicles/hour)")
    ax.grid(axis="y", alpha=0.25)

    path = output_dir / "weekday_vs_weekend.png"
    save_figure(fig, path, "weekday versus weekend traffic")

    if len(grouped) == 2:
        weekday_value = float(grouped.loc[grouped["day_type"] == "Weekday", "traffic_volume"].iloc[0])
        weekend_value = float(grouped.loc[grouped["day_type"] == "Weekend", "traffic_volume"].iloc[0])
        logger.info(
            "Weekday/weekend interpretation: weekday average %.2f vs weekend average %.2f vehicles/hour",
            weekday_value,
            weekend_value,
        )


def plot_temperature_vs_traffic(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot temperature in Celsius against traffic volume."""
    temp_c = pd.to_numeric(df["temp"], errors="coerce") - 273.15
    traffic = pd.to_numeric(df["traffic_volume"], errors="coerce")

    plot_df = pd.DataFrame({"temperature_c": temp_c, "traffic_volume": traffic}).dropna()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(plot_df["temperature_c"], plot_df["traffic_volume"], alpha=0.35, s=12)
    ax.set_title("Temperature vs Traffic Volume")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Traffic Volume (vehicles/hour)")
    ax.grid(alpha=0.25)

    path = output_dir / "temperature_vs_traffic.png"
    save_figure(fig, path, "temperature and traffic relationship")

    correlation = plot_df["temperature_c"].corr(plot_df["traffic_volume"])
    anomaly_count = int((plot_df["temperature_c"] < -100).sum())
    logger.info(
        "Temperature/traffic interpretation: Pearson correlation %.4f; %d observations are below -100°C and should be treated as anomalies",
        correlation,
        anomaly_count,
    )


def create_visualisations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create all required visualisations."""
    plot_traffic_by_hour(df, output_dir)
    plot_weekday_vs_weekend(df, output_dir)
    plot_temperature_vs_traffic(df, output_dir)
    logger.info("All visualisations created successfully")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Matplotlib traffic visualisations.")
    parser.add_argument("--input", type=Path, required=True, help="Feature-engineered CSV path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for figures")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/pipeline.log"),
        help="Combined log file path",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    configure_logging(args.log, getattr(logging, args.log_level))
    logger.info("Visualisation task started")

    try:
        df = load_features(args.input)
        create_visualisations(df, args.output_dir)
        logger.info("Visualisation task completed successfully")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Visualisation task failed: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # entry-point safety net
        logger.error("Unexpected visualisation failure: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
