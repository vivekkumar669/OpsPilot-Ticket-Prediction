import numpy as np
import pandas as pd
from pathlib import Path

from src.prediction.data_loader import load_daily_data
from src.prediction.sla_forecast import forecast_sla


def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def calculate_wape(actual, predicted):

    total_actual = np.sum(np.abs(actual))

    if total_actual == 0:
        return np.nan

    return (
        np.sum(np.abs(actual - predicted))
        / total_actual
    ) * 100


def run_backtest(df, horizon, evaluation_days):

    methods = [
        "weekday_average",
        "gradient_boosting"
    ]

    first_origin = len(df) - evaluation_days

    step = horizon

    all_results = []

    for method in methods:

        actual_evaluations = []
        predicted_evaluations = []

        actual_breaches = []
        predicted_breaches = []

        rate_errors = []

        origins = range(
            first_origin,
            len(df) - horizon + 1,
            step
        )

        for origin in origins:

            # Only use data available at this historical date
            history = df.iloc[:origin].copy()

            # Actual future observations
            actual_future = df.iloc[
                origin:origin + horizon
            ]

            # Forecast future SLA metrics
            predictions, summary = forecast_sla(
                history_df=history,
                horizon=horizon,
                method=method
            )

            actual_eval = (
                actual_future[
                    "sla_evaluations_recorded"
                ].to_numpy()
            )

            actual_breach = (
                actual_future[
                    "sla_breaches_recorded"
                ].to_numpy()
            )

            predicted_eval = (
                predictions[
                    "predicted_sla_evaluations"
                ].to_numpy()
            )

            predicted_breach = (
                predictions[
                    "predicted_sla_breaches"
                ].to_numpy()
            )

            actual_evaluations.extend(actual_eval)
            predicted_evaluations.extend(predicted_eval)

            actual_breaches.extend(actual_breach)
            predicted_breaches.extend(predicted_breach)

            # Compare aggregate SLA breach rate
            actual_total_eval = actual_eval.sum()
            actual_total_breach = actual_breach.sum()

            predicted_total_eval = predicted_eval.sum()
            predicted_total_breach = predicted_breach.sum()

            if (
                actual_total_eval > 0
                and predicted_total_eval > 0
            ):

                actual_rate = (
                    actual_total_breach
                    / actual_total_eval
                ) * 100

                predicted_rate = (
                    predicted_total_breach
                    / predicted_total_eval
                ) * 100

                rate_errors.append(
                    abs(actual_rate - predicted_rate)
                )

        actual_evaluations = np.array(actual_evaluations)
        predicted_evaluations = np.array(predicted_evaluations)

        actual_breaches = np.array(actual_breaches)
        predicted_breaches = np.array(predicted_breaches)

        all_results.append({

            "Horizon": horizon,
            "Model": method,

            "Evaluation_MAE": round(
                calculate_mae(
                    actual_evaluations,
                    predicted_evaluations
                ), 3
            ),

            "Evaluation_WAPE": round(
                calculate_wape(
                    actual_evaluations,
                    predicted_evaluations
                ), 2
            ),

            "Breach_MAE": round(
                calculate_mae(
                    actual_breaches,
                    predicted_breaches
                ), 3
            ),

            "Breach_WAPE": round(
                calculate_wape(
                    actual_breaches,
                    predicted_breaches
                ), 2
            ),

            "Breach_Rate_MAE_PP": round(
                np.mean(rate_errors), 3
            )
        })

    return all_results


# ------------------------------------
# Main execution
# ------------------------------------

df = load_daily_data()

results = []

# Seven-day backtesting across the last 140 days
results.extend(
    run_backtest(
        df,
        horizon=7,
        evaluation_days=140
    )
)

# Thirty-day backtesting across the last 180 days
results.extend(
    run_backtest(
        df,
        horizon=30,
        evaluation_days=180
    )
)

results_df = pd.DataFrame(results)

print("\nSLA BACKTEST RESULTS")

print(
    results_df.to_string(index=False)
)

# Save results
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "sla_backtest_results.csv",
    index=False
)

print("\nResults saved to outputs/sla_backtest_results.csv")