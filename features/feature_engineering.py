import pandas as pd
import glob
import os
import logging

# ----------------------------
# CONFIG
# ----------------------------
SHIPPING_PATH = "data/raw/shipping/*.csv"
NEWS_PATH = "data/raw/news/*.csv"
GEO_PATH = "data/raw/geo/*.csv"

OUTPUT_PATH = "data/processed/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# LOAD LATEST FILE
# ----------------------------
def load_latest_file(path_pattern):
    files = glob.glob(path_pattern)
    if not files:
        logging.warning(f"No files found for {path_pattern}")
        return None

    latest_file = max(files, key=os.path.getctime)
    logging.info(f"Loading {latest_file}")
    return pd.read_csv(latest_file)


# ----------------------------
# SHIPPING FEATURES
# ----------------------------
def process_shipping(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Rolling features
    df["freight_ma_7"] = df["freight_rate"].rolling(7).mean()
    df["freight_volatility"] = df["freight_rate"].rolling(7).std()

    df["congestion_ma_7"] = df["congestion_index"].rolling(7).mean()

    # FIX HERE
    df.bfill(inplace=True)

    return df


# ----------------------------
# NEWS FEATURES
# ----------------------------
def process_news(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Basic feature: news count per day
    news_count = df.groupby("date").size().reset_index(name="news_count")

    return news_count


# ----------------------------
# GEO FEATURES
# ----------------------------
def process_geo(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Aggregate per day
    geo_daily = df.groupby("date").agg({
        "geo_risk_score": "mean"
    }).reset_index()

    return geo_daily


# ----------------------------
# MERGE ALL FEATURES
# ----------------------------
def merge_features(ship_df, news_df, geo_df):
    logging.info("Merging features...")

    df = ship_df.merge(news_df, on="date", how="left")
    df = df.merge(geo_df, on="date", how="left")

    df.fillna(0, inplace=True)

    return df


# ----------------------------
# SAVE FINAL DATASET
# ----------------------------
def save_data(df):
    filepath = os.path.join(OUTPUT_PATH, "final_features.csv")
    df.to_csv(filepath, index=False)

    logging.info(f"Saved processed data to {filepath}")


# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline():
    ship_df = load_latest_file(SHIPPING_PATH)
    news_df = load_latest_file(NEWS_PATH)
    geo_df = load_latest_file(GEO_PATH)

    if ship_df is None:
        logging.error("Shipping data missing. Cannot proceed.")
        return

    ship_df = process_shipping(ship_df)

    if news_df is not None:
        news_df = process_news(news_df)
    else:
        news_df = pd.DataFrame(columns=["date", "news_count"])

    if geo_df is not None:
        geo_df = process_geo(geo_df)
    else:
        geo_df = pd.DataFrame(columns=["date", "geo_risk_score"])

    final_df = merge_features(ship_df, news_df, geo_df)

    save_data(final_df)


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_pipeline()