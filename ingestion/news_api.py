import requests
import pandas as pd
from datetime import datetime
import logging
import os

# ----------------------------
# CONFIG
# ----------------------------
NEWS_API_KEY = "YOUR_API_KEY"  # replace later
NEWS_URL = "https://newsapi.org/v2/everything"

SAVE_PATH = "data/raw/news/"
os.makedirs(SAVE_PATH, exist_ok=True)

KEYWORDS = [
    "port strike",
    "supply chain disruption",
    "factory shutdown",
    "semiconductor shortage",
    "logistics delay",
    "shipping crisis"
]

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# FETCH NEWS
# ----------------------------
def fetch_news():
    """
    Fetch news articles from NewsAPI
    """
    try:
        logging.info("Fetching news data...")

        query = " OR ".join(KEYWORDS)

        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50
        }

        response = requests.get(NEWS_URL, params=params)

        if response.status_code != 200:
            logging.warning("Using simulated news data (API failed)")
            return simulate_news()

        articles = response.json().get("articles", [])

        if not articles:
            logging.warning("No articles found, using simulated data")
            return simulate_news()

        data = []
        for article in articles:
            data.append({
                "date": article.get("publishedAt"),
                "title": article.get("title"),
                "description": article.get("description"),
                "source": article.get("source", {}).get("name")
            })

        df = pd.DataFrame(data)
        logging.info(f"Fetched {len(df)} articles")

        return df

    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return simulate_news()


# ----------------------------
# SIMULATED DATA (Fallback)
# ----------------------------
def simulate_news():
    logging.info("Generating simulated news data...")

    n = 20  # unified size

    titles = [
        "Port strike disrupts shipping in Asia",
        "Factory shutdown impacts semiconductor supply",
        "Logistics delays expected due to weather"
    ]

    data = {
        "date": pd.date_range(end=datetime.today(), periods=n),
        "title": (titles * (n // len(titles) + 1))[:n],  # repeat safely
        "description": ["Supply chain disruption observed"] * n,
        "source": ["Reuters"] * n
    }

    return pd.DataFrame(data)


# ----------------------------
# CLEAN DATA
# ----------------------------
def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess news data
    """
    try:
        logging.info("Cleaning news data...")

        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df.dropna(subset=["title"], inplace=True)

        df.fillna("", inplace=True)

        df.sort_values("date", ascending=False, inplace=True)

        return df

    except Exception as e:
        logging.error(f"Error cleaning news: {e}")
        return df


# ----------------------------
# SAVE DATA
# ----------------------------
def save_data(df: pd.DataFrame):
    try:
        filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(SAVE_PATH, filename)

        df.to_csv(filepath, index=False)

        logging.info(f"Saved news data to {filepath}")

    except Exception as e:
        logging.error(f"Error saving news data: {e}")


# ----------------------------
# PIPELINE
# ----------------------------
def run_pipeline():
    df = fetch_news()

    if df is not None:
        df = clean_news(df)
        save_data(df)
    else:
        logging.warning("No news data available")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    run_pipeline()