import pandas as pd
import logging
import os
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/processed/final_features.csv"
MODEL_OUTPUT = "models/artifacts/"
os.makedirs(MODEL_OUTPUT, exist_ok=True)

TARGET_COLUMN = "freight_rate"

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# LOAD DATA
# ----------------------------
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


# ----------------------------
# TRAIN ARIMA
# ----------------------------
def train_arima(series):
    try:
        logging.info("Training ARIMA model...")

        # (p,d,q) - simple starting config
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()

        logging.info("Model training complete")
        return model_fit

    except Exception as e:
        logging.error(f"Error training ARIMA: {e}")
        return None


# ----------------------------
# FORECAST
# ----------------------------
def forecast(model_fit, steps=14):
    try:
        logging.info(f"Forecasting next {steps} steps...")

        forecast = model_fit.forecast(steps=steps)

        return forecast

    except Exception as e:
        logging.error(f"Error forecasting: {e}")
        return None


# ----------------------------
# SAVE FORECAST
# ----------------------------
def save_forecast(forecast):
    try:
        df = pd.DataFrame({
            "forecast": forecast
        })

        filepath = os.path.join(MODEL_OUTPUT, "arima_forecast.csv")
        df.to_csv(filepath)

        logging.info(f"Saved forecast to {filepath}")

    except Exception as e:
        logging.error(f"Error saving forecast: {e}")


# ----------------------------
# PLOT (OPTIONAL BUT USEFUL)
# ----------------------------
def plot_forecast(series, forecast):
    plt.figure()
    plt.plot(series[-50:], label="Actual")
    plt.plot(range(len(series), len(series) + len(forecast)), forecast, label="Forecast")
    plt.legend()
    plt.title("ARIMA Forecast")
    plt.show()


# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline():
    df = load_data()

    if df is None:
        logging.error("No data found")
        return

    series = df[TARGET_COLUMN]

    model_fit = train_arima(series)

    if model_fit is None:
        return

    fc = forecast(model_fit, steps=14)

    if fc is not None:
        save_forecast(fc)
        plot_forecast(series, fc)


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_pipeline()