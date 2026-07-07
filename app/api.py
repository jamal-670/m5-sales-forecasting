from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from m5_forecasting.models.xgboost import (
    build_xgboost_model,
    train_xgboost_model,
    predict_xgboost_model,
)

app = FastAPI(title="M5 Forecasting API", version="1.0.0")


class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = pd.DataFrame(
        [{"feature_1": request.feature_1, "feature_2": request.feature_2}]
    )

    model = build_xgboost_model(n_estimators=25, max_depth=3, random_state=42)
    trained_model = train_xgboost_model(
        model,
        pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature_2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            }
        ),
        pd.Series([3.0, 6.0, 9.0, 12.0, 15.0, 18.0]),
    )
    prediction = predict_xgboost_model(trained_model, features)[0]
    return {"prediction": float(prediction)}
