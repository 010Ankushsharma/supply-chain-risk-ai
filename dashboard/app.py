import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import requests

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_PATH = "data/processed/final_features.csv"
ARIMA_PATH = "models/artifacts/arima_forecast.csv"
LSTM_PATH = "models/artifacts/lstm_forecast.csv"

df = pd.read_csv(DATA_PATH)
arima = pd.read_csv(ARIMA_PATH)
lstm = pd.read_csv(LSTM_PATH)

# ----------------------------
# INIT DASH APP
# ----------------------------
app = dash.Dash(__name__)
app.title = "Supply Chain Risk Dashboard"

# ----------------------------
# LAYOUT
# ----------------------------
app.layout = html.Div([
    html.H1("Supply Chain Risk Monitoring System"),

    # ----------------------------
    # Freight Trend
    # ----------------------------
    html.H2("Freight Rate Trend"),
    dcc.Graph(
        figure={
            "data": [
                go.Scatter(
                    x=df["date"],
                    y=df["freight_rate"],
                    mode="lines",
                    name="Freight Rate"
                )
            ],
            "layout": go.Layout(title="Freight Rate Over Time")
        }
    ),

    # ----------------------------
    # Forecast Comparison
    # ----------------------------
    html.H2("Forecast Comparison"),
    dcc.Graph(
        figure={
            "data": [
                go.Scatter(
                    y=arima["forecast"],
                    mode="lines",
                    name="ARIMA Forecast"
                ),
                go.Scatter(
                    y=lstm["forecast"],
                    mode="lines",
                    name="LSTM Forecast"
                )
            ],
            "layout": go.Layout(title="Forecast (Next Steps)")
        }
    ),

    # ----------------------------
    # Risk Prediction Button
    # ----------------------------
    html.H2("Risk Prediction"),
    html.Button("Get Risk Prediction", id="predict-btn", n_clicks=0),
    html.Div(id="prediction-output")
])

# ----------------------------
# CALLBACK
# ----------------------------
@app.callback(
    dash.dependencies.Output("prediction-output", "children"),
    [dash.dependencies.Input("predict-btn", "n_clicks")]
)
def get_prediction(n_clicks):
    if n_clicks == 0:
        return ""

    # Take latest row as input
    latest = df.iloc[-1]

    payload = {
        "freight_rate": float(latest["freight_rate"]),
        "freight_ma_7": float(latest["freight_ma_7"]),
        "freight_volatility": float(latest["freight_volatility"]),
        "congestion_ma_7": float(latest["congestion_ma_7"]),
        "news_count": float(latest["news_count"]),
        "geo_risk_score": float(latest["geo_risk_score"]),
        "arima_forecast_mean": float(arima["forecast"].mean()),
        "lstm_forecast_mean": float(lstm["forecast"].mean())
    }

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = res.json()

        return html.Div([
            html.H3(f"Risk Level: {result['risk_level']}"),
            html.P(f"Confidence: {round(result['confidence'], 2)}")
        ])

    except Exception as e:
        return f"Error: {str(e)}"


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)