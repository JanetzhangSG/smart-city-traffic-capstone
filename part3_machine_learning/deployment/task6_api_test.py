from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from task6_api import create_app

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Task 6 FastAPI deployment mock")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", handlers=[logging.FileHandler(args.log, encoding="utf-8"), logging.StreamHandler()])
    app = create_app(args.model)
    client = TestClient(app)
    health = client.get("/health")
    LOGGER.info("Health response: %s", health.json())
    payload = {
        "holiday": "None",
        "temp": 290.0,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 20.0,
        "weather_main": "Clear",
        "weather_description": "sky is clear",
        "date_time": datetime(2017, 8, 1, 16, 0, 0).isoformat(),
    }
    response = client.post("/predict", json=payload)
    response.raise_for_status()
    LOGGER.info("Prediction response: %s", response.json())
    print(response.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
