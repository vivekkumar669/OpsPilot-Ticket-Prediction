
from pathlib import Path
import joblib

from src.prediction.data_loader import load_daily_data
from src.prediction.forecast import train_gradient_boosting

# Project directories
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load historical data
df = load_daily_data()

# Train ticket arrival model
ticket_model = train_gradient_boosting(
    df,
    target_col="incoming_tickets"
)

joblib.dump(ticket_model, MODEL_DIR / "ticket_forecast.joblib")

# Train SLA evaluation model
evaluation_model = train_gradient_boosting(
    df,
    target_col="sla_evaluations_recorded"
)

joblib.dump(
    evaluation_model,
    MODEL_DIR / "sla_evaluations.joblib"
)

# Train SLA breach model
breach_model = train_gradient_boosting(
    df,
    target_col="sla_breaches_recorded"
)

joblib.dump(
    breach_model,
    MODEL_DIR / "sla_breaches.joblib"
)

print("MODELS TRAINED AND SAVED SUCCESSFULLY")
print("Saved models in:", MODEL_DIR)

