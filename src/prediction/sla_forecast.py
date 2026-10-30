
import numpy as np
import pandas as pd

from src.prediction.forecast import (
    weekday_average_forecast_multi,
    train_gradient_boosting,
    forecast_gradient_boosting_multi
)


def forecast_sla(
    history_df,
    horizon=7,
    method="weekday_average"
):
    """
    Forecast SLA evaluations and SLA breaches.

    Methods:
        weekday_average
        gradient_boosting

    Returns:
        Daily predictions and aggregate SLA summary.
    """

    required_columns = [
        "date",
        "sla_evaluations_recorded",
        "sla_breaches_recorded"
    ]

    missing = set(required_columns) - set(history_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if history_df.empty:
        raise ValueError("Historical data is empty")

    if horizon < 1:
        raise ValueError("Horizon must be at least 1")

    history = history_df.sort_values("date").copy()

    if history["date"].duplicated().any():
        raise ValueError("Duplicate dates found")

    if (
        history["sla_evaluations_recorded"] < 0
    ).any() or (
        history["sla_breaches_recorded"] < 0
    ).any():
        raise ValueError("SLA counts cannot be negative")

    if (
        history["sla_breaches_recorded"]
        > history["sla_evaluations_recorded"]
    ).any():
        raise ValueError(
            "SLA breaches cannot exceed SLA evaluations"
        )

    if len(history) < 29:
        raise ValueError(
            "At least 29 days of history are required"
        )

    targets = [
        "sla_evaluations_recorded",
        "sla_breaches_recorded"
    ]

    predicted_counts = {}

    # Forecast both SLA count series separately
    for target_col in targets:

        if method == "weekday_average":

            result = weekday_average_forecast_multi(
                history,
                horizon=horizon,
                target_col=target_col
            )

        elif method == "gradient_boosting":

            model = train_gradient_boosting(
                history,
                target_col=target_col
            )

            result = forecast_gradient_boosting_multi(
                model,
                history,
                horizon=horizon,
                target_col=target_col
            )

        else:
            raise ValueError(
                "Method must be 'weekday_average' "
                "or 'gradient_boosting'"
            )

        predicted_counts[target_col] = (
            result["predicted_tickets"].to_numpy()
        )

    # Convert predicted counts into non-negative integers
    evaluations = np.maximum(
        np.rint(
            predicted_counts["sla_evaluations_recorded"]
        ),
        0
    ).astype(int)

    breaches = np.maximum(
        np.rint(
            predicted_counts["sla_breaches_recorded"]
        ),
        0
    ).astype(int)

    # A breach cannot exceed the number of evaluations
    breaches = np.minimum(breaches, evaluations)

    # Daily breach percentage
    daily_rates = np.divide(
        breaches * 100.0,
        evaluations,
        out=np.full(len(evaluations), np.nan),
        where=evaluations > 0
    )

    future_dates = pd.date_range(
        start=history["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    daily_predictions = pd.DataFrame({
        "date": future_dates,
        "predicted_sla_evaluations": evaluations,
        "predicted_sla_breaches": breaches,
        "predicted_sla_breach_rate_pct": daily_rates
    })

    total_evaluations = int(evaluations.sum())
    total_breaches = int(breaches.sum())

    overall_breach_rate = (
        total_breaches / total_evaluations * 100
        if total_evaluations > 0
        else np.nan
    )

    summary = {
        "horizon_days": horizon,
        "method": method,
        "predicted_evaluations": total_evaluations,
        "predicted_breaches": total_breaches,
        "predicted_breach_rate_pct": round(
            overall_breach_rate, 2
        ) if not np.isnan(overall_breach_rate) else None
    }

    return daily_predictions, summary
