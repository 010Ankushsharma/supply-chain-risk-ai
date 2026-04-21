import requests
import pandas as pd
from datetime import datetime
import logging
import os

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "https://api.example.com/freight"  # replace later
SAVE_PATH = "data/raw/shipping/"
os.makedirs(SAVE_PATH, exist_ok=True)

# ----------------------------
# LOGGING SETUP
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# FETCH DATA
# ----------------------------
def fetch_shipping_data():
    """
    Fetch shipping data from API
    """
    try:
        logging.info("Fetching shipping data...")

        # TEMP: simulate API response (replace later)
        data = {
            "date": pd.date_range(end=datetime.today(), periods=30),
            "freight_rate": [1000 + i * 5 for i in range(30)],
            "congestion_index": [50 + i * 0.5 for i in range(30)],
            "lead_time": [10 + i * 0.2 for i in range(30)]
        }

        df = pd.DataFrame(data)

        logging.info(f"Fetched {len(df)} records")
        return df

    except Exception as e:
        logging.error(f"Error fetching shipping data: {e}")
        return None


# ----------------------------
# CLEAN DATA
# ----------------------------
def clean_shipping_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess shipping data
    """
    try:
        logging.info("Cleaning shipping data...")

        df = df.copy()

        # Convert date
        df["date"] = pd.to_datetime(df["date"])

        # Handle missing values
        df.fillna(method="ffill", inplace=True)

        # Sort by date
        df.sort_values("date", inplace=True)

        return df

    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        return df


# ----------------------------
# SAVE DATA
# ----------------------------
def save_data(df: pd.DataFrame):
    """
    Save data to CSV (later DB)
    """
    try:
        filename = f"shipping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(SAVE_PATH, filename)

        df.to_csv(filepath, index=False)

        logging.info(f"Saved data to {filepath}")

    except Exception as e:
        logging.error(f"Error saving data: {e}")


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def run_pipeline():
    """
    Full ingestion pipeline
    """
    df = fetch_shipping_data()

    if df is not None:
        df = clean_shipping_data(df)
        save_data(df)
    else:
        logging.warning("No data fetched.")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_pipeline()