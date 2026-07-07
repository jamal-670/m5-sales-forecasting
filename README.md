# M5 Sales Forecasting

This repository contains an end-to-end forecasting pipeline for the Walmart M5 Sales Forecasting challenge. It includes data loading, feature engineering, baseline and advanced model experimentation, model persistence, and evaluation utilities.

## Project structure

- data/raw/: raw Kaggle files such as calendar, sales, and sell prices
- src/m5_forecasting/data/: dataset loading and preprocessing helpers
- src/m5_forecasting/features/: time, lag, rolling, and other feature engineering functions
- src/m5_forecasting/models/: model definitions and training/evaluation utilities for LSTM and XGBoost
- notebooks/: exploratory analysis and model development notebooks
- reports/: generated predictions, figures, and results

## Included workflows

- Data loading from the M5 competition files
- Feature engineering for time-based and lag-based signals
- Baseline forecasting experiments
- XGBoost regression modeling
- LSTM model support for sequence-based forecasting
- Model persistence and evaluation with MAE, RMSE, and MAPE

## Installation

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The M5 Forecasting dataset is not included because of its size.

Download it from the Kaggle competition:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Place the files in:

```bash
data/raw/
```

## Example usage

Train and evaluate an XGBoost model:

```python
from pathlib import Path
import pandas as pd
from m5_forecasting.models.xgboost import (
    build_xgboost_model,
    train_xgboost_model,
    predict_xgboost_model,
    save_xgboost_model,
)
from m5_forecasting.models.evaluator import evaluate_model

X = pd.DataFrame({
    "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "feature_2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
})
y = pd.Series([3.0, 6.0, 9.0, 12.0, 15.0, 18.0])

model = build_xgboost_model(n_estimators=10, max_depth=2, random_state=42)
trained_model = train_xgboost_model(model, X, y)
preds = predict_xgboost_model(trained_model, X)
metrics = evaluate_model(y, preds)

save_xgboost_model(trained_model, Path("models/xgboost_model.json"))
print(metrics)
```

## Docker deployment

Build and run the app with Docker Compose:

```bash
docker compose up --build
```

Then open http://localhost:8501 in your browser.

## Simple frontend

A lightweight Streamlit interface is available in [app/streamlit_app.py](app/streamlit_app.py). It lets you:

- train a sample XGBoost model
- inspect MAE, RMSE, and MAPE directly in the browser
- call the live FastAPI prediction endpoint from a form

A Hugging Face Spaces-compatible entry point is also available in [app/app.py](app/app.py).

Run it locally with:

```bash
streamlit run app/streamlit_app.py
```

For Hugging Face Spaces with Docker, use:

```bash
docker build -f Dockerfile.hf -t m5-forecasting-demo .
docker run -p 8501:8501 m5-forecasting-demo
```

## FastAPI backend

A simple REST API is available in [app/api.py](app/api.py). It exposes:

- GET /health for a basic health check
- POST /predict for generating a forecast from a small feature payload

Run it locally with:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Free-hosting deployment

This project is now set up for free-friendly deployment using the FastAPI service as the main app.

### Render deployment

Use [render.yaml](render.yaml) and [runtime.txt](runtime.txt) to deploy on Render for free.

1. Push the repository to GitHub.
2. Create a new Render web service from the repo.
3. Render will detect [render.yaml](render.yaml) and start the app with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The root URL will show a simple landing page, and the API will be available under /api.

### Local optional UI

If you want to run the Streamlit demo locally, use:

```bash
streamlit run app/streamlit_app.py
```

## Notes

The project is designed to be extended with additional models and richer forecasting pipelines as more data and experiments are added.