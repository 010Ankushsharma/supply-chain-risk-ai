import pandas as pd
import logging
import os

from transformers import pipeline

# ----------------------------
# CONFIG
# ----------------------------
NEWS_PATH = "data/raw/news/*.csv"
OUTPUT_PATH = "data/processed/news_nlp.csv"

# ----------------------------
# LOAD MODEL
# ----------------------------
logging.info("Loading FinBERT model...")
sentiment_model = pipeline("sentiment-analysis")

# ----------------------------
# LOAD DATA
# ----------------------------
def load_news():
    import glob
    files = glob.glob(NEWS_PATH)
    latest = max(files, key=os.path.getctime)
    return pd.read_csv(latest)


# ----------------------------
# PROCESS NLP
# ----------------------------
def process_nlp(df):
    sentiments = []

    for text in df["title"]:
        try:
            res = sentiment_model(text[:512])[0]
            sentiments.append(res["score"])
        except:
            sentiments.append(0)

    df["sentiment_score"] = sentiments

    # Aggregate daily sentiment
    agg = df.groupby("date").agg({
        "sentiment_score": "mean"
    }).reset_index()

    return agg


# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline():
    df = load_news()
    df["date"] = pd.to_datetime(df["date"])

    nlp_df = process_nlp(df)
    nlp_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved NLP features")


if __name__ == "__main__":
    run_pipeline()