
import pandas as pd

from src.prediction.data_loader import load_daily_data

from src.prediction.forecast import (
    weekday_average_forecast,
    train_gradient_boosting,
    predict_gradient_boosting,
)

from src.prediction.evaluation import evaluate_model
from pathlib import Path

# --------------------------------
# 1. Load historical data
# --------------------------------

df = load_daily_data()

print("\nDATA LOADED")
print("Total records:", len(df))


# --------------------------------
# 2. Create chronological split
# --------------------------------

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[validation_end:].copy()

print("\nDATA SPLIT")

print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# --------------------------------
# 3. Train Gradient Boosting
# --------------------------------

# At this stage, fit on training data only.
model = train_gradient_boosting(train_df)

print("\nGRADIENT BOOSTING TRAINED")


# --------------------------------
# 4. Prepare test history
# --------------------------------

# Earlier observations are available when
# making predictions for future test dates.
history_df = df.iloc[:validation_end].copy()

test_history = pd.concat(
    [history_df, test_df],
    ignore_index=True
)


# --------------------------------
# 5. Baseline predictions
# --------------------------------

baseline_predictions = weekday_average_forecast(
    test_history
)

baseline_predictions = baseline_predictions.iloc[
    validation_end:
].to_numpy()


# --------------------------------
# 6. Gradient Boosting predictions
# --------------------------------

prediction_dates, all_ml_predictions = (
    predict_gradient_boosting(
        model,
        test_history
    )
)

# Keep only test-period predictions.
ml_predictions = all_ml_predictions[-len(test_df):]

actual = test_df["incoming_tickets"].to_numpy()


# --------------------------------
# 7. Evaluate models
# --------------------------------

baseline_results = evaluate_model(
    actual,
    baseline_predictions
)

ml_results = evaluate_model(
    actual,
    ml_predictions
)


# --------------------------------
# 8. Display comparison
# --------------------------------

results = pd.DataFrame(
    [
        {
            "Model": "Four-Week Weekday Average",
            **baseline_results
        },
        {
            "Model": "HistGradientBoosting",
            **ml_results
        }
    ]
)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))


# --------------------------------
# 9. Best model
# --------------------------------

best_model = results.loc[
    results["MAE"].idxmin()
]

print("\nBEST MODEL BASED ON MAE:")
print(best_model["Model"])


# --------------------------------
# 10. Save predictions
# --------------------------------

# Create outputs directory if it doesn't exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save prediction comparison
output_path = OUTPUT_DIR / "forecast_comparison.csv"

output.to_csv(
    output_path,
    index=False
)

print(f"\nPredictions saved to: {output_path}")