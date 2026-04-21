🚀 Supply Chain Risk Intelligence System

An end-to-end AI-powered supply chain risk prediction system that forecasts disruptions 4–8 weeks in advance using time-series modeling, NLP, and machine learning.

📌 Overview

Global supply chains are highly vulnerable to disruptions such as port strikes, geopolitical conflicts, and logistics delays. Most organizations react only after disruptions occur.

This system enables proactive risk management by:

Monitoring shipping, news, and geopolitical data
Forecasting future supply chain conditions
Predicting disruption risk levels (Low / Medium / High)
Providing explainable insights for decision-making
🧠 Key Features
📥 Multi-source Data Ingestion
Shipping data (freight rates, congestion, lead time)
News data (global supply chain events)
Geopolitical risk data
🧠 NLP Pipeline
Sentiment analysis using transformer models
Converts unstructured news into risk signals
📊 Hybrid Time-Series Forecasting
ARIMA → linear trends
LSTM → nonlinear patterns
⚠️ Risk Classification Model
XGBoost classifier
Predicts disruption severity
🔍 Explainability
SHAP-based feature importance
🌐 API Layer
FastAPI-based model serving
📊 Interactive Dashboard
Built with Plotly Dash
🔔 Alert System
Slack + Email notifications
🐳 Dockerized Deployment
Fully containerized multi-service system
🏗️ Architecture
Data Sources → Ingestion → Feature Engineering → Models → API → Dashboard → Alerts
📁 Project Structure
supply-chain-risk-ai/
│
├── ingestion/
│   ├── shipping_api.py
│   ├── news_api.py
│   ├── geo_api.py
│
├── features/
│   ├── feature_engineering.py
│   ├── nlp_pipeline.py
│
├── models/
│   ├── arima_model.py
│   ├── lstm_model.py
│   ├── xgboost_model.py
│
├── explainability/
│   ├── shap_explainer.py
│
├── api/
│   ├── main.py
│
├── dashboard/
│   ├── app.py
│
├── alerts/
│   ├── slack_alert.py
│   ├── email_alert.py
│
├── config/
│   ├── config.yaml
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
⚙️ Tech Stack
Layer	Tools
Data Ingestion	Python, Requests
NLP	Transformers (FinBERT), spaCy
Time Series	statsmodels (ARIMA), PyTorch/TensorFlow (LSTM)
ML Model	XGBoost
Explainability	SHAP
Backend	FastAPI
Dashboard	Plotly Dash
Database	PostgreSQL (optional)
Deployment	Docker Compose
🚀 Getting Started
1️⃣ Setup Environment
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
2️⃣ Run Data Pipelines
python ingestion/shipping_api.py
python ingestion/news_api.py
python ingestion/geo_api.py
3️⃣ Run NLP Pipeline
python features/nlp_pipeline.py
4️⃣ Feature Engineering
python features/feature_engineering.py
5️⃣ Train Models
python models/arima_model.py
python models/lstm_model.py
python models/xgboost_model.py
6️⃣ Start API
uvicorn api.main:app --reload

👉 Open: http://127.0.0.1:8000/docs

7️⃣ Run Dashboard
python dashboard/app.py

👉 Open: http://127.0.0.1:8050

8️⃣ Run with Docker (Optional)
docker-compose up --build
📊 Sample API Request
{
  "freight_rate": 1050,
  "freight_ma_7": 1020,
  "freight_volatility": 15,
  "congestion_ma_7": 60,
  "news_count": 5,
  "geo_risk_score": 2,
  "arima_forecast_mean": 1100,
  "lstm_forecast_mean": 1080
}
📤 Sample API Response
{
  "risk_level": "HIGH",
  "confidence": 0.89
}
📈 Outputs
Forecasts:
arima_forecast.csv
lstm_forecast.csv
Model:
xgboost_model.json
Explainability:
shap_summary.png
🧪 Future Improvements
MLflow for experiment tracking
Airflow for pipeline orchestration
Real-time streaming data ingestion
Advanced NLP (entity-level risk mapping)
Supplier-level risk scoring
💡 Key Learnings
Designing end-to-end ML systems
Combining time-series + NLP + tabular ML
Building production-ready pipelines
Model explainability in decision systems
Deploying ML as a service
📜 License

This project is for educational and research purposes.

🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

👨‍💻 Author

Ankush Sharma
Aspiring ML Engineer | AI Systems Builder