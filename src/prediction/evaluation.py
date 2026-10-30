
import numpy as np


def calculate_mae(actual, predicted):
    """
    Mean Absolute Error.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return np.mean(
        np.abs(actual - predicted)
    )


def calculate_wape(actual, predicted):
    """
    Weighted Absolute Percentage Error.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    total_actual = np.sum(np.abs(actual))

    if total_actual == 0:
        return float("nan")

    error = np.sum(
        np.abs(actual - predicted)
    )

    return (error / total_actual) * 100


def evaluate_model(actual, predicted):

    return {
        "MAE": calculate_mae(actual, predicted),
        "WAPE": calculate_wape(actual, predicted)
    }
