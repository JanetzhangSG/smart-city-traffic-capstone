#!/usr/bin/env python3
"""Task 5 - Traffic Recommendation System.

A single-corridor timing recommender based on historical traffic volume,
day type, and optional weather condition. It does not select routes and does
not claim to predict actual accidents.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend lower-traffic travel windows.")
    parser.add_argument("--input", required=True, help="Path to cleaned traffic CSV")
    parser.add_argument("--output-dir", default=".", help="Output project directory")
    parser.add_argument("--day-type", choices=["Weekday", "Weekend", "Any"], default="Weekday")
    parser.add_argument("--weather", default="Any", help="Weather condition or Any")
    parser.add_argument("--window-hours", type=int, default=1, choices=[1, 2], help="Recommendation window length")
    parser.add_argument("--top-n", type=int, default=3, help="Number of recommendations")
    parser.add_argument("--log", default="logs/task5.log", help="Log file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"date_time", "traffic_volume", "weather_main", "holiday"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(df)
    out = df.copy()
    out["date_time"] = pd.to_datetime(out["date_time"], errors="coerce")
    bad_dates = int(out["date_time"].isna().sum())
    if bad_dates:
        LOGGER.warning("Dropped %d rows with invalid date_time values", bad_dates)
        out = out.dropna(subset=["date_time"])

    out["hour"] = out["date_time"].dt.hour
    out["day_of_week"] = out["date_time"].dt.dayofweek
    out["day_type"] = out["day_of_week"].apply(lambda d: "Weekend" if d >= 5 else "Weekday")
    out["weather_main"] = out["weather_main"].astype(str).str.strip().replace({"nan": "Unknown"})
    out["traffic_volume"] = pd.to_numeric(out["traffic_volume"], errors="coerce")
    out = out.dropna(subset=["traffic_volume"])
    LOGGER.info("Prepared recommendation dataset: %d rows", len(out))
    return out


def apply_filters(df: pd.DataFrame, day_type: str, weather: str) -> pd.DataFrame:
    filtered = df
    if day_type != "Any":
        filtered = filtered[filtered["day_type"] == day_type]
    if weather.lower() != "any":
        filtered = filtered[filtered["weather_main"].str.lower() == weather.lower()]

    if filtered.empty:
        raise ValueError("No observations match the selected day type and weather condition.")

    LOGGER.info(
        "Applied filters: day_type=%s weather=%s -> %d observations",
        day_type, weather, len(filtered)
    )
    return filtered


def hourly_profile(df: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    hourly = (
        df.groupby("hour", as_index=False)["traffic_volume"]
        .agg(avg_traffic="mean", observations="count")
        .sort_values("hour")
    )

    rows = []
    for start in range(0, 24 - window_hours + 1):
        hours = list(range(start, start + window_hours))
        subset = hourly[hourly["hour"].isin(hours)]
        if len(subset) != window_hours:
            continue
        weighted_avg = (subset["avg_traffic"] * subset["observations"]).sum() / subset["observations"].sum()
        rows.append({
            "start_hour": start,
            "end_hour": start + window_hours - 1,
            "window": f"{start:02d}:00-{(start + window_hours) % 24:02d}:00",
            "avg_traffic": weighted_avg,
            "observations": int(subset["observations"].sum()),
        })
    return pd.DataFrame(rows).sort_values("avg_traffic")


def make_recommendations(filtered: pd.DataFrame, day_type: str, weather: str, window_hours: int, top_n: int) -> pd.DataFrame:
    profile = hourly_profile(filtered, window_hours).head(top_n).copy()
    if profile.empty:
        raise ValueError("Unable to form recommendation windows from the filtered observations.")

    conditions = f"{day_type.lower()}" if day_type != "Any" else "selected"
    if weather.lower() != "any":
        conditions += f" under {weather.title()} weather"

    profile["recommendation"] = profile.apply(
        lambda r: (
            f"For a {conditions} journey, consider travelling between {r['window']}, "
            f"when historical traffic volumes average about {r['avg_traffic']:.0f} vehicles/hour."
        ),
        axis=1,
    )
    return profile


def save_outputs(recs: pd.DataFrame, filtered: pd.DataFrame, output_dir: Path, day_type: str, weather: str) -> None:
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "task5_recommendations.csv"
    json_path = results_dir / "task5_recommendations.json"
    profile_path = results_dir / "task5_filtered_hourly_profile.csv"
    fig_path = figures_dir / "task5_recommended_windows.png"
    txt_path = results_dir / "task5_recommendation_summary.txt"

    recs.to_csv(csv_path, index=False)
    recs.to_json(json_path, orient="records", indent=2)
    filtered.groupby("hour", as_index=False)["traffic_volume"].mean().rename(columns={"traffic_volume": "avg_traffic"}).to_csv(profile_path, index=False)

    summary = [
        "Task 5 - Traffic Recommendation System",
        f"Day type: {day_type}",
        f"Weather: {weather}",
        "Note: the dataset represents a single corridor, so recommendations concern travel timing rather than route choice.",
        "",
    ]
    for i, row in recs.reset_index(drop=True).iterrows():
        summary.append(f"{i + 1}. {row['recommendation']}")
    txt_path.write_text("\n".join(summary), encoding="utf-8")

    # Visualise the filtered hourly profile and highlight recommended windows.
    hourly = filtered.groupby("hour", as_index=False)["traffic_volume"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(hourly["hour"], hourly["traffic_volume"], marker="o", linewidth=1.5, label="Average traffic")
    for _, row in recs.iterrows():
        plt.axvspan(row["start_hour"], row["end_hour"] + 1, alpha=0.18)
    plt.title(f"Recommended Travel Windows - {day_type}, {weather}")
    plt.xlabel("Hour of day")
    plt.ylabel("Average traffic volume (vehicles/hour)")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    LOGGER.info("Saved recommendations: %s", csv_path)
    LOGGER.info("Saved recommendation summary: %s", txt_path)
    LOGGER.info("Figure saved: %s", fig_path)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOGGER.addHandler(file_handler)

    try:
        LOGGER.info("Part 3 Task 5 traffic recommendation started")
        df = pd.read_csv(args.input)
        LOGGER.info("Loaded recommendation input: %d rows x %d columns", len(df), len(df.columns))
        prepared = prepare(df)
        filtered = apply_filters(prepared, args.day_type, args.weather)
        recs = make_recommendations(filtered, args.day_type, args.weather, args.window_hours, args.top_n)
        for i, row in recs.reset_index(drop=True).iterrows():
            LOGGER.info("Recommendation %d: %s", i + 1, row["recommendation"])
        save_outputs(recs, filtered, Path(args.output_dir), args.day_type, args.weather)
        LOGGER.info("Part 3 Task 5 completed successfully")
        # Direct user-facing CLI output is acceptable; internal status remains logged.
        print(recs[["window", "avg_traffic", "recommendation"]].to_string(index=False))
        return 0
    except (FileNotFoundError, pd.errors.ParserError, ValueError, OSError) as exc:
        LOGGER.error("Task 5 recommendation failed: %s", exc, exc_info=True)
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
