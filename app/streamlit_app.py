import streamlit as st
import pandas as pd
import requests
from pathlib import Path

from m5_forecasting.models.xgboost import (
    build_xgboost_model,
    train_xgboost_model,
    predict_xgboost_model,
    save_xgboost_model,
)
from m5_forecasting.models.evaluator import evaluate_model

st.set_page_config(page_title="M5 Forecasting App", page_icon="📈", layout="wide")

st.title("M5 Sales Forecasting")
st.write("A simple forecasting demo with a trained XGBoost model and a live API endpoint.")

api_url = st.text_input("API base URL", value="http://localhost:8000")

with st.sidebar:
    st.header("Model training")
    n_estimators = st.slider("Number of trees", 10, 200, 50)
    max_depth = st.slider("Max depth", 2, 10, 4)
    train_button = st.button("Train demo model")

if train_button:
    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    y = pd.Series([3.0, 6.0, 9.0, 12.0, 15.0, 18.0])

    model = build_xgboost_model(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    trained_model = train_xgboost_model(model, X, y)
    predictions = predict_xgboost_model(trained_model, X)
    metrics = evaluate_model(y, predictions)

    save_path = Path("models/xgboost_streamlit_model.json")
    save_xgboost_model(trained_model, save_path)

    st.success(f"Demo model trained and saved to {save_path}")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(metrics["mae"], 3))
    col2.metric("RMSE", round(metrics["rmse"], 3))
    col3.metric("MAPE", round(metrics["mape"], 3))

    st.dataframe(pd.DataFrame({"actual": y, "prediction": predictions}))

st.header("Live prediction")
feature_1 = st.number_input("Feature 1", value=4.0)
feature_2 = st.number_input("Feature 2", value=8.0)

if st.button("Predict"):
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"feature_1": feature_1, "feature_2": feature_2},
            timeout=10,
        )
        if response.ok:
            payload = response.json()
            st.metric("Prediction", round(payload["prediction"], 3))
        else:
            st.error(f"API request failed: {response.status_code}")
    except Exception as exc:
        st.error(f"Could not connect to the API: {exc}")

st.info("Use the API endpoint at /predict for deployment or testing.")
