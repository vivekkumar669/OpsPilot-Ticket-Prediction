
from pathlib import Path

import pandas as pd


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default dataset location
DATA_PATH = PROJECT_ROOT / "data" / "daily_metrics_730_days.csv"


def load_daily_data(file_path=DATA_PATH):

    # 1. Load CSV
    df = pd.read_csv(file_path)

    # 2. Required columns
    required_columns = [
        "date",
        "incoming_tickets",
        "resolved_tickets",
        "open_backlog_end_of_day",
        "sla_evaluations_recorded",
        "sla_breaches_recorded",
    ]

    # 3. Check required columns
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # 4. Convert date column
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError("Invalid dates found")

    # 5. Sort data by date
    df = df.sort_values("date").reset_index(drop=True)

    # 6. Check duplicate dates
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    # 7. Check missing values
    if df[required_columns].isna().any().any():
        raise ValueError("Missing values found")

    # 8. Convert numeric columns
    numeric_columns = [
        "incoming_tickets",
        "resolved_tickets",
        "open_backlog_end_of_day",
        "sla_evaluations_recorded",
        "sla_breaches_recorded",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_columns].isna().any().any():
        raise ValueError("Invalid numeric values found")

    # 9. Check non-negative values
    if (df[numeric_columns] < 0).any().any():
        raise ValueError("Negative values found")

    # 10. SLA breaches cannot exceed evaluations
    if (
        df["sla_breaches_recorded"]
        > df["sla_evaluations_recorded"]
    ).any():
        raise ValueError(
            "SLA breaches exceed SLA evaluations"
        )

    # 11. Check missing calendar dates
    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D"
    )

    if not df["date"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="date")
    ):
        raise ValueError("Missing calendar dates found")

    return df
