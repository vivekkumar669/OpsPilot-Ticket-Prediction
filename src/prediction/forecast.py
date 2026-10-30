

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from src.prediction.features import (
    create_features,
    get_feature_columns
)


def weekday_average_forecast(df):
    """
    Predict each day's ticket arrivals using
    the average of the previous four matching weekdays.
    """

    data = df.copy().reset_index(drop=True)

    predictions = (
        data["incoming_tickets"].shift(7)
        + data["incoming_tickets"].shift(14)
        + data["incoming_tickets"].shift(21)
        + data["incoming_tickets"].shift(28)
    ) / 4

    return predictions



def train_gradient_boosting(
    train_df,
    target_col="incoming_tickets"
):
    """
    Train HistGradientBoostingRegressor
    for the selected target column.
    """

    feature_df = create_features(
        train_df,
        target_col=target_col
    )

    feature_columns = get_feature_columns()

    X_train = feature_df[feature_columns]
    y_train = feature_df[target_col]

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model



def predict_gradient_boosting(model, history_df):
    """
    Generate one-step-ahead predictions
    using the historical values available
    for each prediction date.
    """

    feature_df = create_features(
        history_df,
        target_col="incoming_tickets"
    )

    feature_columns = get_feature_columns()

    X = feature_df[feature_columns]

    predictions = model.predict(X)

    # Ticket arrivals cannot be negative.
    predictions = np.maximum(predictions, 0)

    return feature_df["date"], predictions


def weekday_average_forecast_multi(
    history_df,
    horizon=7,
    target_col="incoming_tickets"
):
    """
    Recursive forecast using the average of
    the previous four matching weekdays.
    """

    if horizon < 1:
        raise ValueError("Horizon must be at least 1")

    history = history_df.sort_values("date").copy()

    values = history[target_col].astype(float).tolist()
    last_date = history["date"].iloc[-1]

    if len(values) < 28:
        raise ValueError("At least 28 historical days are required")

    if history["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    predictions = []

    for step in range(horizon):

        # Same weekday from the previous four weeks
        recent_same_weekdays = [
            values[-7],
            values[-14],
            values[-21],
            values[-28]
        ]

        predicted_value = float(np.mean(recent_same_weekdays))

        predicted_value = max(0.0, predicted_value)

        future_date = last_date + pd.Timedelta(days=step + 1)

        predictions.append({
            "date": future_date,
            "predicted_tickets": predicted_value
        })

        # Feed the prediction into future history
        values.append(predicted_value)

    return pd.DataFrame(predictions)


def forecast_gradient_boosting_multi(
    model,
    history_df,
    horizon=7,
    target_col="incoming_tickets"
):
    """
    Recursive multi-step forecasting using
    HistGradientBoostingRegressor.
    """

    if horizon < 1:
        raise ValueError("Horizon must be at least 1")

    history = history_df.sort_values("date").copy()

    values = history[target_col].astype(float).tolist()
    last_date = history["date"].iloc[-1]

    if len(values) < 28:
        raise ValueError("At least 28 historical days are required")

    if history["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    feature_columns = get_feature_columns()

    lags = (1, 7, 14, 21, 28)
    rolling_windows = (7, 14, 28)

    predictions = []

    for step in range(horizon):

        future_date = last_date + pd.Timedelta(days=step + 1)

        features = {
            "day_of_week": future_date.dayofweek,
            "day_of_month": future_date.day,
            "month": future_date.month,
            "is_weekend": int(future_date.dayofweek >= 5)
        }

        # Create lag features
        for lag in lags:
            features[f"lag_{lag}"] = values[-lag]

        # Create rolling average features
        for window in rolling_windows:
            features[f"rolling_mean_{window}"] = float(
                np.mean(values[-window:])
            )

        X_future = pd.DataFrame(
            [features],
            columns=feature_columns
        )

        predicted_value = float(model.predict(X_future)[0])

        # Counts cannot be negative
        predicted_value = max(0.0, predicted_value)

        predictions.append({
            "date": future_date,
            "predicted_tickets": predicted_value
        })

        # Add prediction for the next iteration
        values.append(predicted_value)

    return pd.DataFrame(predictions)
