import pandas as pd
from datetime import datetime
import logging
import os

# ----------------------------
# CONFIG
# ----------------------------
SAVE_PATH = "data/raw/geo/"
os.makedirs(SAVE_PATH, exist_ok=True)

# Example file path (later replace with real ACLED/GPR data)
LOCAL_FILE = "data/external/acled_sample.csv"

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# FETCH DATA
# ----------------------------
def fetch_geo_data():
    """
    Fetch geopolitical data (CSV or simulated)
    """
    try:
        logging.info("Fetching geopolitical data...")

        if os.path.exists(LOCAL_FILE):
            df = pd.read_csv(LOCAL_FILE)
            logging.info("Loaded ACLED data from local file")
        else:
            logging.warning("No local file found, using simulated data")
            df = simulate_geo_data()

        return df

    except Exception as e:
        logging.error(f"Error fetching geo data: {e}")
        return simulate_geo_data()


# ----------------------------
# SIMULATED DATA
# ----------------------------
def simulate_geo_data():
    """
    Generate fake geopolitical events
    """
    logging.info("Generating simulated geopolitical data...")

    data = {
        "date": pd.date_range(end=datetime.today(), periods=30),
        "country": ["China", "USA", "India", "Germany"] * 7 + ["China", "USA"],
        "event_type": ["Protest", "Conflict", "Strike", "Violence"] * 7 + ["Conflict", "Strike"],
        "fatalities": [0, 5, 0, 2] * 7 + [3, 1]
    }

    return pd.DataFrame(data)


# ----------------------------
# CLEAN + FEATURE ENGINEERING
# ----------------------------
def process_geo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and aggregate geopolitical data
    """
    try:
        logging.info("Processing geopolitical data...")

        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        # Basic aggregation: events per country per day
        agg_df = (
            df.groupby(["date", "country"])
            .agg({
                "event_type": "count",
                "fatalities": "sum"
            })
            .reset_index()
        )

        agg_df.rename(columns={
            "event_type": "event_count"
        }, inplace=True)

        # Risk score (simple heuristic)
        agg_df["geo_risk_score"] = (
            agg_df["event_count"] * 0.7 +
            agg_df["fatalities"] * 0.3
        )

        return agg_df

    except Exception as e:
        logging.error(f"Error processing geo data: {e}")
        return df


# ----------------------------
# SAVE DATA
# ----------------------------
def save_data(df: pd.DataFrame):
    try:
        filename = f"geo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(SAVE_PATH, filename)

        df.to_csv(filepath, index=False)

        logging.info(f"Saved geo data to {filepath}")

    except Exception as e:
        logging.error(f"Error saving geo data: {e}")


# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline():
    df = fetch_geo_data()

    if df is not None:
        df = process_geo_data(df)
        save_data(df)
    else:
        logging.warning("No geo data available")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_pipeline()