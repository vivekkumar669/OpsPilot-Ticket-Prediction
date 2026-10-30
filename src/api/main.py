
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prediction.forecast import (
    weekday_average_forecast_multi,
    forecast_gradient_boosting_multi
)

from src.prediction.sla_forecast import forecast_sla


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="OpsPilot Engine API",
    description="Prediction and operational analysis engine",
    version="1.0.0"
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


ticket_model = load_model("ticket_forecast.joblib")
sla_evaluation_model = load_model("sla_evaluations.joblib")
sla_breach_model = load_model("sla_breaches.joblib")


# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------

class ForecastRequest(BaseModel):
    history: list[dict]
    horizon: int = Field(default=7, ge=1, le=30)
    method: str = "auto"


# --------------------------------------------------
# HELPER
# --------------------------------------------------

def prepare_history(history):
    if not history:
        raise ValueError("Historical data cannot be empty")

    df = pd.DataFrame(history)

    if "date" not in df.columns:
        raise ValueError("Missing date column")

    df["date"] = pd.to_datetime(df["date"])

    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "OpsPilot Engine"
    }


# --------------------------------------------------
# TICKET FORECAST
# --------------------------------------------------

@app.post("/forecast")
def forecast(request: ForecastRequest):

    try:
        df = prepare_history(request.history)

        if "incoming_tickets" not in df.columns:
            raise ValueError("Missing incoming_tickets column")

        if df["incoming_tickets"].isna().any():
            raise ValueError("Ticket history contains missing values")

        if (df["incoming_tickets"] < 0).any():
            raise ValueError("Ticket counts cannot be negative")

        if df["date"].duplicated().any():
            raise ValueError("Duplicate dates found")

        if len(df) < 29:
            raise ValueError("At least 29 days of history required")

        method = request.method

        if method == "auto":
            method = (
                "weekday_average"
                if request.horizon == 7
                else "gradient_boosting"
            )

        if method == "weekday_average":

            result = weekday_average_forecast_multi(
                df,
                horizon=request.horizon,
                target_col="incoming_tickets"
            )

        elif method == "gradient_boosting":

            result = forecast_gradient_boosting_multi(
                ticket_model,
                df,
                horizon=request.horizon,
                target_col="incoming_tickets"
            )

        else:
            raise ValueError("Invalid forecasting method")

        result["date"] = result["date"].dt.strftime("%Y-%m-%d")

        return {
            "status": "success",
            "horizon_days": request.horizon,
            "method": method,
            "predictions": result.to_dict(orient="records")
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


# --------------------------------------------------
# SLA FORECAST
# --------------------------------------------------

@app.post("/sla-forecast")
def sla_forecast(request: ForecastRequest):

    try:
        df = prepare_history(request.history)

        method = request.method

        if method == "auto":
            method = (
                "gradient_boosting"
                if request.horizon == 7
                else "gradient_boosting"
            )

        if method not in ["weekday_average", "gradient_boosting"]:
            raise ValueError("Invalid SLA forecasting method")

        # Use saved models instead of retraining.
        # Temporarily use the existing SLA function for
        # weekday-average forecasts only.
        if method == "weekday_average":
            daily, summary = forecast_sla(
                df,
                horizon=request.horizon,
                method="weekday_average"
            )
        else:
            # Build both forecasts with saved models.
            evaluations = forecast_gradient_boosting_multi(
                sla_evaluation_model,
                df,
                horizon=request.horizon,
                target_col="sla_evaluations_recorded"
            )

            breaches = forecast_gradient_boosting_multi(
                sla_breach_model,
                df,
                horizon=request.horizon,
                target_col="sla_breaches_recorded"
            )

            import numpy as np

            evaluation_counts = np.maximum(
                np.rint(evaluations["predicted_tickets"]), 0
            ).astype(int)

            breach_counts = np.maximum(
                np.rint(breaches["predicted_tickets"]), 0
            ).astype(int)

            breach_counts = np.minimum(
                breach_counts, evaluation_counts
            )

            rates = np.divide(
                breach_counts * 100.0,
                evaluation_counts,
                out=np.zeros(len(evaluation_counts)),
                where=evaluation_counts > 0
            )

            daily = pd.DataFrame({
                "date": evaluations["date"],
                "predicted_sla_evaluations": evaluation_counts,
                "predicted_sla_breaches": breach_counts,
                "predicted_sla_breach_rate_pct": rates
            })

            total_evaluations = int(evaluation_counts.sum())
            total_breaches = int(breach_counts.sum())

            summary = {
                "horizon_days": request.horizon,
                "method": method,
                "predicted_evaluations": total_evaluations,
                "predicted_breaches": total_breaches,
                "predicted_breach_rate_pct": round(
                    total_breaches / total_evaluations * 100, 2
                ) if total_evaluations else None
            }

        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

        return {
            "status": "success",
            "daily_predictions": daily.to_dict(orient="records"),
            "summary": summary
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
