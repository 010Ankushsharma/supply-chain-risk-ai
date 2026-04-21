import numpy as np
import pandas as pd
import logging
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/processed/final_features.csv"
MODEL_OUTPUT = "models/artifacts/"
os.makedirs(MODEL_OUTPUT, exist_ok=True)

TARGET_COLUMN = "freight_rate"
TIME_STEPS = 10  # sequence length

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
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


# ----------------------------
# CREATE SEQUENCES
# ----------------------------
def create_sequences(data, time_steps=10):
    X, y = [], []

    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])

    return np.array(X), np.array(y)


# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess(df):
    scaler = MinMaxScaler()

    values = df[[TARGET_COLUMN]].values
    scaled = scaler.fit_transform(values)

    X, y = create_sequences(scaled, TIME_STEPS)

    return X, y, scaler


# ----------------------------
# BUILD MODEL
# ----------------------------
def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    return model


# ----------------------------
# TRAIN MODEL
# ----------------------------
def train_model(X, y):
    logging.info("Training LSTM model...")

    model = build_model((X.shape[1], X.shape[2]))

    model.fit(X, y, epochs=10, batch_size=8, verbose=1)

    return model


# ----------------------------
# FORECAST
# ----------------------------
def forecast(model, last_sequence, scaler, steps=14):
    preds = []

    current_seq = last_sequence.copy()

    for _ in range(steps):
        pred = model.predict(current_seq.reshape(1, TIME_STEPS, 1), verbose=0)
        preds.append(pred[0][0])

        current_seq = np.append(current_seq[1:], pred)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    return preds.flatten()


# ----------------------------
# SAVE OUTPUT
# ----------------------------
def save_forecast(preds):
    df = pd.DataFrame({"forecast": preds})

    filepath = os.path.join(MODEL_OUTPUT, "lstm_forecast.csv")
    df.to_csv(filepath, index=False)

    logging.info(f"Saved LSTM forecast to {filepath}")


# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline():
    df = load_data()

    if df is None:
        return

    X, y, scaler = preprocess(df)

    model = train_model(X, y)

    last_sequence = X[-1]

    preds = forecast(model, last_sequence, scaler, steps=14)

    save_forecast(preds)


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_pipeline()