from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

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

CATEGORICAL_COLUMNS = ["holiday", "weather_main", "weather_description"]
NUMERIC_COLUMNS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]


def configure_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure console and file logging for the pipeline entry point."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the input schema before any cleaning operations."""
    LOGGER.info("Validating input schema before cleaning")
    actual = list(df.columns)
    missing = [column for column in EXPECTED_COLUMNS if column not in actual]
    extra = [column for column in actual if column not in EXPECTED_COLUMNS]

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if extra:
        LOGGER.warning("Unexpected columns detected: %s", extra)

    LOGGER.info(
        "Schema validation passed: all %d expected columns are present",
        len(EXPECTED_COLUMNS),
    )


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV with explicit error handling."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parsing failed: {csv_path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read raw CSV: {csv_path}") from exc

    LOGGER.info("Raw file loaded successfully: %d rows x %d columns", *df.shape)
    validate_schema(df)
    return df


def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize categorical text values and make holiday missingness explicit."""
    holiday_missing_count = 0
    categorical_value_changes = 0

    for column in CATEGORICAL_COLUMNS:
        original = df[column].copy()

        # Convert non-null values to trimmed strings while preserving missing values.
        cleaned = original.astype("string").str.strip()
        cleaned = cleaned.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA})

        if column == "holiday":
            missing_mask = cleaned.isna()
            holiday_missing_count += int(missing_mask.sum())
            # Blank holiday values mean no holiday in this dataset; make that explicit.
            cleaned = cleaned.fillna("None")

        df[column] = cleaned

        # Count genuine standardization changes only; holiday blank -> None is logged separately.
        if column == "holiday":
            comparable_original = original.astype("string").str.strip().replace(
                {"": pd.NA, "<NA>": pd.NA, "nan": pd.NA}
            )
            comparable_new = df[column].astype("string")
            genuine_mask = (
                comparable_original.notna()
                & (comparable_original != comparable_new)
            )
        else:
            comparable_original = original.astype("string")
            comparable_new = df[column].astype("string")
            genuine_mask = comparable_original.fillna("<MISSING>") != comparable_new.fillna("<MISSING>")

        categorical_value_changes += int(genuine_mask.sum())

    if holiday_missing_count:
        LOGGER.warning(
            "Standardized %d blank holiday entries to 'None' to represent non-holiday observations",
            holiday_missing_count,
        )
    else:
        LOGGER.info("No blank holiday entries required standardization")

    if categorical_value_changes:
        LOGGER.warning(
            "Standardized %d categorical values for whitespace/inconsistent text",
            categorical_value_changes,
        )
    else:
        LOGGER.info("No additional categorical values required standardization")

    LOGGER.info("Categorical standardization completed")
    return df


def parse_and_validate_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time and drop rows that cannot be parsed."""
    parsed = pd.to_datetime(df["date_time"], errors="coerce")
    invalid_count = int(parsed.isna().sum())

    if invalid_count:
        df = df.loc[parsed.notna()].copy()
        parsed = parsed.loc[parsed.notna()]
        LOGGER.warning(
            "Dropped %d rows because date_time could not be parsed",
            invalid_count,
        )
    else:
        LOGGER.info("Date/time parsing completed; 0 values failed validation")

    df["date_time"] = parsed
    if invalid_count:
        LOGGER.info(
            "Date/time parsing completed; %d valid rows retained",
            len(df),
        )
    return df


def coerce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns use numeric dtypes and report newly created missing values."""
    total_new_missing = 0
    for column in NUMERIC_COLUMNS:
        before_missing = int(df[column].isna().sum())
        converted = pd.to_numeric(df[column], errors="coerce")
        after_missing = int(converted.isna().sum())
        new_missing = max(after_missing - before_missing, 0)
        if new_missing:
            LOGGER.warning(
                "Converted %d non-numeric %s values to missing values during numeric validation",
                new_missing,
                column,
            )
            total_new_missing += new_missing
        df[column] = converted

    LOGGER.info("Numeric type validation completed")
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows."""
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        df = df.loc[~duplicate_mask].copy()
        LOGGER.warning("Dropped %d duplicate rows", duplicate_count)
    else:
        LOGGER.info("Duplicate check completed; no duplicate rows found")
    return df


def monthly_median(df: pd.DataFrame, column: str) -> pd.Series:
    """Return the median for each observation's calendar month."""
    month_medians = df.groupby(df["date_time"].dt.month)[column].transform("median")
    return month_medians


def impute_outliers_by_month(
    df: pd.DataFrame,
    mask: pd.Series,
    column: str,
    reason: str,
) -> int:
    """Replace flagged values with the median for the observation's month."""
    count = int(mask.sum())
    if count == 0:
        LOGGER.info("No %s detected", reason)
        return 0

    monthly_values = monthly_median(df, column)
    replacement = monthly_values.loc[mask]
    fallback = df.loc[mask, column].median()
    replacement = replacement.fillna(fallback)

    # Explicit loop keeps the group-by-month application visible as requested by the assignment.
    for index in replacement.index:
        df.loc[index, column] = replacement.loc[index]

    LOGGER.warning("Imputed %d %s using monthly medians", count, reason)
    return count


def detect_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect impossible/extreme sensor values and impute them by month."""
    temp_mask = (df["temp"] <= 0) | (df["temp"] > 330)
    rain_mask = (df["rain_1h"] < 0) | (df["rain_1h"] > 9)
    snow_mask = (df["snow_1h"] < 0) | (df["snow_1h"] > 2)
    clouds_mask = (df["clouds_all"] < 0) | (df["clouds_all"] > 100)
    traffic_mask = df["traffic_volume"] < 0

    total_changes = 0
    total_changes += impute_outliers_by_month(
        df, temp_mask, "temp", "physically implausible temperature values"
    )
    total_changes += impute_outliers_by_month(
        df, rain_mask, "rain_1h", "rain_1h values outside the plausible range"
    )
    total_changes += impute_outliers_by_month(
        df, snow_mask, "snow_1h", "snow_1h values outside the plausible range"
    )

    if clouds_mask.any():
        affected = int(clouds_mask.sum())
        median_value = df.loc[~clouds_mask, "clouds_all"].median()
        for index in df.index[clouds_mask]:
            df.loc[index, "clouds_all"] = median_value
        LOGGER.warning("Imputed %d impossible clouds_all values using the median", affected)
        total_changes += affected
    else:
        LOGGER.info("No impossible clouds_all values detected")

    if traffic_mask.any():
        affected = int(traffic_mask.sum())
        median_value = df.loc[~traffic_mask, "traffic_volume"].median()
        for index in df.index[traffic_mask]:
            df.loc[index, "traffic_volume"] = median_value
        LOGGER.warning(
            "Imputed %d negative traffic_volume values using the median",
            affected,
        )
        total_changes += affected
    else:
        LOGGER.info("No impossible traffic_volume values detected")

    LOGGER.info(
        "Outlier/impossible-value handling completed; %d values affected",
        total_changes,
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete cleaning sequence."""
    LOGGER.info("Cleaning step 1/5: standardizing categorical values")
    df = standardize_categorical_values(df)

    LOGGER.info("Cleaning step 2/5: parsing and validating date/time")
    df = parse_and_validate_datetime(df)

    LOGGER.info("Cleaning step 3/5: validating numeric columns")
    df = coerce_numeric_types(df)

    LOGGER.info("Cleaning step 4/5: removing duplicate rows")
    df = remove_duplicate_rows(df)

    LOGGER.info("Cleaning step 5/5: detecting and handling outliers/impossible values")
    df = detect_and_handle_outliers(df)

    LOGGER.info("Cleaning pipeline completed: %d rows remain", len(df))
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Cleaned dataset saved: %s", output_path)


def run_pipeline(csv_path: Path, cleaned_path: Path, log_path: Path) -> int:
    """Run the pipeline and exit gracefully on expected/unexpected errors."""
    configure_logging(log_path)
    LOGGER.info("Pipeline started")

    try:
        raw = load_raw_csv(csv_path)
        cleaned = clean_data(raw.copy())
        save_cleaned_data(cleaned, cleaned_path)
        LOGGER.info("Pipeline completed successfully")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("Pipeline failed: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # final safety net for graceful exit
        LOGGER.error("Unexpected pipeline failure: %s", exc, exc_info=True)
        return 1


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
    cleaned_path = project_root / "part2_python" / "outputs" / "traffic_cleaned.csv"
    log_path = project_root / "part2_python" / "logs" / "pipeline.log"
    return run_pipeline(csv_path, cleaned_path, log_path)


if __name__ == "__main__":
    sys.exit(main())
