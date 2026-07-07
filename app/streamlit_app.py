import streamlit as st
import pandas as pd
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
st.write("Train a simple XGBoost baseline and inspect its evaluation metrics.")

with st.sidebar:
    st.header("Configuration")
    n_estimators = st.slider("Number of trees", 10, 200, 50)
    max_depth = st.slider("Max depth", 2, 10, 4)
    run_button = st.button("Train model")

if run_button:
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

    st.success(f"Model trained and saved to {save_path}")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(metrics["mae"], 3))
    col2.metric("RMSE", round(metrics["rmse"], 3))
    col3.metric("MAPE", round(metrics["mape"], 3))

    st.dataframe(pd.DataFrame({"actual": y, "prediction": predictions}))
else:
    st.info("Use the sidebar to train a sample forecasting model.")
