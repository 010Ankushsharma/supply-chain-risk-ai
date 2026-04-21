import pandas as pd
import numpy as np
import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "models/artifacts/xgboost_model.json"

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = XGBClassifier()
model.load_model(MODEL_PATH)

# ----------------------------
# INIT APP
# ----------------------------
app = FastAPI(title="Supply Chain Risk API")

# ----------------------------
# INPUT SCHEMA
# ----------------------------
class InputData(BaseModel):
    freight_rate: float
    freight_ma_7: float
    freight_volatility: float
    congestion_ma_7: float
    news_count: float
    geo_risk_score: float
    arima_forecast_mean: float
    lstm_forecast_mean: float


# ----------------------------
# HELPER: RISK LABEL
# ----------------------------
def get_risk_label(pred):
    return ["LOW", "MEDIUM", "HIGH"][int(pred)]


# ----------------------------
# ROOT
# ----------------------------
@app.get("/")
def home():
    return {"message": "Supply Chain Risk API Running"}


# ----------------------------
# PREDICT ENDPOINT
# ----------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = data.dict()

        df = pd.DataFrame([input_dict])

        pred = model.predict(df)[0]
        prob = model.predict_proba(df).max()

        return {
            "risk_level": get_risk_label(pred),
            "confidence": float(prob),
            "input": input_dict
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}