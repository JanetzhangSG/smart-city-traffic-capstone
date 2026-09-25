from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

SEVERE_WEATHER = {"Thunderstorm", "Snow", "Squall", "Fog"}
LOW_VISIBILITY_WEATHER = {"Fog", "Mist", "Haze", "Smoke", "Squall"}


class TrafficRequest(BaseModel):
    holiday: str = "None"
    temp: float
    rain_1h: float = 0.0
    snow_1h: float = 0.0
    clouds_all: float = 0.0
    weather_main: str = "Clear"
    weather_description: str = "sky is clear"
    date_time: datetime


def build_features(payload: TrafficRequest) -> pd.DataFrame:
    ts = pd.Timestamp(payload.date_time)
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month
    weather = str(payload.weather_main).strip()
    row = pd.DataFrame(
        [{
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": int(dow >= 5),
            "month": month,
            "is_holiday": int(str(payload.holiday).strip().lower() != "none"),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "day_sin": np.sin(2 * np.pi * dow / 7.0),
            "day_cos": np.cos(2 * np.pi * dow / 7.0),
            "weather_main": weather,
            "weather_description": payload.weather_description,
            "rain_1h": payload.rain_1h,
            "snow_1h": payload.snow_1h,
            "clouds_all": payload.clouds_all,
            "is_low_visibility": int(weather in LOW_VISIBILITY_WEATHER),
            "is_severe_weather": int(weather in SEVERE_WEATHER),
            "has_precipitation": int((payload.rain_1h > 0) or (payload.snow_1h > 0)),
            "temp": payload.temp,
        }]
    )
    return row


def create_app(model_path: Path) -> FastAPI:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    app = FastAPI(title="Smart City Traffic Prediction API", version="1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": "task1-v1-hist_gradient_boosting_regressor"}

    @app.post("/predict")
    def predict(payload: TrafficRequest) -> dict[str, object]:
        try:
            features = build_features(payload)
            prediction = float(model.predict(features)[0])
            LOGGER.info("API prediction generated for %s: %.2f vehicles/hour", payload.date_time.isoformat(), prediction)
            return {
                "model_version": "task1-v1-hist_gradient_boosting_regressor",
                "prediction_traffic_volume": round(prediction, 2),
                "units": "vehicles/hour",
            }
        except Exception as exc:
            LOGGER.error("API prediction failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=400, detail="Prediction could not be generated") from exc

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI mock deployment for traffic prediction")
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.model), host="0.0.0.0", port=8000)
