import pandas as pd
import numpy as np
import logging
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/processed/final_features.csv"
ARIMA_PATH = "models/artifacts/arima_forecast.csv"
LSTM_PATH = "models/artifacts/lstm_forecast.csv"

MODEL_OUTPUT = "models/artifacts/"
os.makedirs(MODEL_OUTPUT, exist_ok=True)

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
    df = pd.read_csv(DATA_PATH)

    # Load forecasts
    arima = pd.read_csv(ARIMA_PATH)
    lstm = pd.read_csv(LSTM_PATH)

    # Use last forecast values as features
    df["arima_forecast_mean"] = arima["forecast"].mean()
    df["lstm_forecast_mean"] = lstm["forecast"].mean()

    return df


# ----------------------------
# CREATE LABELS (SMART LOGIC)
# ----------------------------
def create_labels(df):
    """
    Create synthetic risk labels
    """
    conditions = [
        (df["congestion_index"] > df["congestion_index"].quantile(0.75)) |
        (df["news_count"] > df["news_count"].quantile(0.75)) |
        (df["geo_risk_score"] > df["geo_risk_score"].quantile(0.75)),

        (df["congestion_index"] > df["congestion_index"].quantile(0.5))
    ]

    choices = [2, 1]  # HIGH, MEDIUM

    df["risk_label"] = np.select(conditions, choices, default=0)

    return df


# ----------------------------
# TRAIN MODEL
# ----------------------------
def train_model(df):
    features = [
        "freight_rate",
        "freight_ma_7",
        "freight_volatility",
        "congestion_ma_7",
        "news_count",
        "geo_risk_score",
        "arima_forecast_mean",
        "lstm_forecast_mean"
    ]

    X = df[features]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(n_estimators=100, max_depth=4)

    logging.info("Training XGBoost model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    logging.info("\n" + classification_report(y_test, preds))

    return model, X


# ----------------------------
# SHAP EXPLAINABILITY
# ----------------------------
def explain_model(model, X):
    logging.info("Generating SHAP explanations...")

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)

    plot_path = os.path.join(MODEL_OUTPUT, "shap_summary.png")
    import matplotlib.pyplot as plt
    plt.savefig(plot_path)

    logging.info(f"Saved SHAP plot to {plot_path}")


# ----------------------------
# SAVE MODEL
# ----------------------------
def save_model(model):
    filepath = os.path.join(MODEL_OUTPUT, "xgboost_model.json")
    model.save_model(filepath)

    logging.info(f"Model saved to {filepath}")


# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline():
    df = load_data()

    df = create_labels(df)

    model, X = train_model(df)

    explain_model(model, X)

    save_model(model)


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_pipeline()