
import numpy as np
import pandas as pd
from pathlib import Path

from src.prediction.data_loader import load_daily_data

from src.prediction.forecast import (
    train_gradient_boosting,
    weekday_average_forecast_multi,
    forecast_gradient_boosting_multi
)

from src.prediction.evaluation import (
    calculate_mae,
    calculate_wape
)


def run_backtest(df, horizon, evaluation_days):

    methods = [
        "weekday_average",
        "gradient_boosting"
    ]

    first_origin = len(df) - evaluation_days

    results = []

    for method in methods:

        all_actual = []
        all_predictions = []

        # Forecast consecutive, non-overlapping windows
        for origin in range(
            first_origin,
            len(df) - horizon + 1,
            horizon
        ):

            # Historical data available at forecast time
            history = df.iloc[:origin].copy()

            # Actual values in the future window
            actual_future = df.iloc[
                origin:origin + horizon
            ]

            # Generate forecast
            if method == "weekday_average":

                forecast = weekday_average_forecast_multi(
                    history,
                    horizon=horizon,
                    target_col="incoming_tickets"
                )

            else:

                model = train_gradient_boosting(
                    history,
                    target_col="incoming_tickets"
                )

                forecast = forecast_gradient_boosting_multi(
                    model,
                    history,
                    horizon=horizon,
                    target_col="incoming_tickets"
                )

            all_actual.extend(
                actual_future["incoming_tickets"].to_numpy()
            )

            all_predictions.extend(
                forecast["predicted_tickets"].to_numpy()
            )

        actual = np.array(all_actual)
        predicted = np.array(all_predictions)

        results.append({
            "Horizon": horizon,
            "Model": method,
            "MAE": round(
                calculate_mae(actual, predicted), 3
            ),
            "WAPE": round(
                calculate_wape(actual, predicted), 2
            ),
            "Forecast_Days_Evaluated": len(actual)
        })

    return results


# ---------------------------------
# Main execution
# ---------------------------------

df = load_daily_data()

results = []

print("\nTICKET ARRIVAL BACKTESTING")

results.extend(
    run_backtest(
        df,
        horizon=7,
        evaluation_days=140
    )
)

results.extend(
    run_backtest(
        df,
        horizon=30,
        evaluation_days=180
    )
)

results_df = pd.DataFrame(results)

print("\nTICKET ARRIVAL BACKTEST RESULTS")

print(
    results_df.to_string(index=False)
)

# Save results
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "ticket_backtest_results.csv",
    index=False
)

print("\nSaved to outputs/ticket_backtest_results.csv")


