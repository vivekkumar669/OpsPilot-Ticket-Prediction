
import pandas as pd


def create_features(
    df,
    target_col="incoming_tickets",
    lags=(1, 7, 14,21, 28),
    rolling_windows=(7, 14, 28)
):
    """
    Create time-series features for forecasting.

    target_col:
        Column we want to predict.

    Returns:
        DataFrame containing features and target.
    """

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found"
        )

    data = df.copy()

    # Ensure chronological order
    data = data.sort_values("date").reset_index(drop=True)

    # Calendar features
    data["day_of_week"] = data["date"].dt.dayofweek

    data["day_of_month"] = data["date"].dt.day

    data["month"] = data["date"].dt.month

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Lag features
    for lag in lags:

        data[f"lag_{lag}"] = (
            data[target_col].shift(lag)
        )

    # Rolling average features
    for window in rolling_windows:

        data[f"rolling_mean_{window}"] = (
            data[target_col]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    # Remove rows where historical features
    # are unavailable
    data = data.dropna().reset_index(drop=True)

    return data


def get_feature_columns(
    lags=(1, 7, 14,21,28),
    rolling_windows=(7, 14, 28)
):
    """
    Return the names of model input features.
    """

    features = [
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend"
    ]

    features.extend(
        [f"lag_{lag}" for lag in lags]
    )

    features.extend(
        [
            f"rolling_mean_{window}"
            for window in rolling_windows
        ]
    )

    return features
