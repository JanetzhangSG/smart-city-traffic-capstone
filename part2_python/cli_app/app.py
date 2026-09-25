"""Command-line traffic analytics application for the capstone project.

The CLI reads the feature-engineered traffic dataset and provides simple,
end-user queries. Internal progress is written with Python logging; print()
is used only for the actual answer returned to the CLI user.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def configure_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure console and file logging for the CLI entry point."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setLevel(level)
        handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False


def load_data(path: Path) -> pd.DataFrame:
    """Load the feature-engineered dataset and validate required fields."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input CSV not found: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parsing failed: {path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read CSV: {path}") from exc

    required = {
        "date_time",
        "hour",
        "day_of_week_name",
        "is_weekend",
        "traffic_volume",
        "weather_main",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    if df["date_time"].isna().any():
        bad = int(df["date_time"].isna().sum())
        raise ValueError(f"Found {bad} invalid date_time values")

    logger.info("Loaded feature-engineered dataset: %d rows x %d columns", *df.shape)
    return df


def _validate_hour(value: str) -> int:
    """Validate an hour argument in the inclusive range 0-23."""
    try:
        hour = int(value)
    except ValueError as exc:
        raise ValueError("Hour must be an integer from 0 to 23") from exc
    if not 0 <= hour <= 23:
        raise ValueError("Hour must be an integer from 0 to 23")
    return hour


def _validate_date(value: str) -> pd.Timestamp:
    """Validate a date argument in YYYY-MM-DD format."""
    timestamp = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("Date must use YYYY-MM-DD format")
    return timestamp


def command_hourly(df: pd.DataFrame, hour_text: str) -> None:
    """Show average traffic for a specified hour of day."""
    hour = _validate_hour(hour_text)
    subset = df[df["hour"] == hour]
    if subset.empty:
        print(f"No observations found for hour {hour:02d}:00.")
        return
    average = float(subset["traffic_volume"].mean())
    print(f"Hour {hour:02d}:00 average traffic: {average:.2f} vehicles/hour")


def command_high_traffic(df: pd.DataFrame, n_text: str) -> None:
    """Show the highest-traffic hours based on historical hourly averages."""
    try:
        n = int(n_text)
    except ValueError as exc:
        raise ValueError("n must be a positive integer") from exc
    if n <= 0:
        raise ValueError("n must be a positive integer")

    hourly = (
        df.groupby("hour", as_index=False)["traffic_volume"]
        .mean()
        .sort_values("traffic_volume", ascending=False)
        .head(n)
    )

    print("Highest-traffic hours:")
    for _, row in hourly.iterrows():
        print(f"  {int(row['hour']):02d}:00 - {row['traffic_volume']:.2f} vehicles/hour")


def command_compare(df: pd.DataFrame) -> None:
    """Compare average traffic on weekdays and weekends."""
    grouped = (
        df.assign(day_type=df["is_weekend"].map({0: "Weekday", 1: "Weekend"}))
        .groupby("day_type", as_index=False)["traffic_volume"]
        .mean()
    )
    print("Average traffic by day type:")
    for _, row in grouped.iterrows():
        print(f"  {row['day_type']}: {row['traffic_volume']:.2f} vehicles/hour")

    if set(grouped["day_type"]) == {"Weekday", "Weekend"}:
        weekday = float(grouped.loc[grouped["day_type"] == "Weekday", "traffic_volume"].iloc[0])
        weekend = float(grouped.loc[grouped["day_type"] == "Weekend", "traffic_volume"].iloc[0])
        difference = weekday - weekend
        print(f"  Weekday minus weekend: {difference:.2f} vehicles/hour")


def command_recommend(df: pd.DataFrame) -> None:
    """Recommend three lower-traffic hours based on historical averages."""
    hourly = (
        df.groupby("hour", as_index=False)["traffic_volume"]
        .mean()
        .sort_values("traffic_volume", ascending=True)
        .head(3)
    )
    print("Recommended lower-traffic travel hours:")
    for _, row in hourly.iterrows():
        print(f"  {int(row['hour']):02d}:00 - {row['traffic_volume']:.2f} vehicles/hour")


def command_weather(df: pd.DataFrame, weather: str) -> None:
    """Show average traffic for a requested weather condition."""
    condition = weather.strip()
    if not condition:
        raise ValueError("weather must not be empty")

    subset = df[df["weather_main"].astype(str).str.casefold() == condition.casefold()]
    if subset.empty:
        available = ", ".join(sorted(df["weather_main"].dropna().astype(str).unique()))
        raise ValueError(f"Unknown weather condition. Available values: {available}")

    average = float(subset["traffic_volume"].mean())
    print(f"{condition.title()} weather average traffic: {average:.2f} vehicles/hour")


def command_date(df: pd.DataFrame, date_text: str) -> None:
    """Show traffic and weather information for a specific calendar date."""
    date = _validate_date(date_text)
    subset = df[df["date_time"].dt.date == date.date()].copy()
    if subset.empty:
        print(f"No observations found for {date_text}.")
        return

    average = float(subset["traffic_volume"].mean())
    maximum = float(subset["traffic_volume"].max())
    weather_values = ", ".join(sorted(subset["weather_main"].dropna().astype(str).unique()))
    print(f"Date: {date_text}")
    print(f"  Observations: {len(subset)}")
    print(f"  Average traffic: {average:.2f} vehicles/hour")
    print(f"  Maximum traffic: {maximum:.0f} vehicles/hour")
    print(f"  Weather conditions: {weather_values}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart City Traffic Analytics CLI")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/traffic_features.csv"),
        help="Feature-engineered CSV path",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/pipeline.log"),
        help="Log file path",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    hourly = subparsers.add_parser("hourly", help="Query average traffic for an hour")
    hourly.add_argument("hour", help="Hour from 0 to 23")

    high = subparsers.add_parser("high-traffic", help="Show the highest-traffic hours")
    high.add_argument("--n", default="5", help="Number of hours to display")

    subparsers.add_parser("compare", help="Compare weekday and weekend traffic")
    subparsers.add_parser("recommend", help="Recommend lower-traffic hours")

    weather = subparsers.add_parser("weather", help="Query average traffic by weather")
    weather.add_argument("condition", help="Weather condition, e.g. Clear")

    date = subparsers.add_parser("date", help="Query traffic and weather for a date")
    date.add_argument("date", help="Date in YYYY-MM-DD format")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log, getattr(logging, args.log_level))

    logger.info("CLI command invoked: %s | arguments=%s", args.command, vars(args))

    try:
        df = load_data(args.input)

        if args.command == "hourly":
            command_hourly(df, args.hour)
        elif args.command == "high-traffic":
            command_high_traffic(df, args.n)
        elif args.command == "compare":
            command_compare(df)
        elif args.command == "recommend":
            command_recommend(df)
        elif args.command == "weather":
            command_weather(df, args.condition)
        elif args.command == "date":
            command_date(df, args.date)
        else:
            raise ValueError(f"Unsupported command: {args.command}")

        logger.info("CLI command completed successfully: %s", args.command)
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("CLI command failed: %s", exc)
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        logger.error("Unexpected CLI failure: %s", exc)
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
