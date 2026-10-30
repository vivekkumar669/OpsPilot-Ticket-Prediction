
from pathlib import Path

import pandas as pd

from src.prediction.data_loader import load_daily_data
from src.prediction.sla_forecast import forecast_sla


# Load historical data
df = load_daily_data()

print("\nHISTORICAL SLA DATA LOADED")
print("Records:", len(df))


output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

summaries = []

# Compare both forecasting approaches
methods = [
    "weekday_average",
    "gradient_boosting"
]

# Generate 7-day and 30-day forecasts
for horizon in [7, 30]:

    print(f"\n{'=' * 50}")
    print(f"{horizon}-DAY SLA FORECAST")
    print(f"{'=' * 50}")

    for method in methods:

        daily_predictions, summary = forecast_sla(
            history_df=df,
            horizon=horizon,
            method=method
        )

        summaries.append(summary)

        print(f"\nMODEL: {method}")

        print(daily_predictions.round(2).to_string(index=False))

        print("\nSUMMARY")

        print(
            "Predicted evaluations:",
            summary["predicted_evaluations"]
        )

        print(
            "Predicted breaches:",
            summary["predicted_breaches"]
        )

        print(
            "Predicted SLA breach rate:",
            summary["predicted_breach_rate_pct"],
            "%"
        )

        # Save individual forecasts
        output_path = (
            output_dir
            / f"sla_forecast_{horizon}_days_{method}.csv"
        )

        daily_predictions.to_csv(
            output_path,
            index=False
        )

        print("Saved:", output_path)


# Save summary comparison
summary_df = pd.DataFrame(summaries)

summary_path = output_dir / "sla_forecast_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nMODEL SUMMARY COMPARISON")
print(summary_df.to_string(index=False))

print("\nSLA FORECASTING COMPLETED")
