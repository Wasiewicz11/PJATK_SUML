import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path

from models import Passenger, PredictionResult
from libs.model import predict

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("ml_models")

app = FastAPI()


@app.get("/", tags=["intro"])
def index():
    return {"message": "Titanic Survival Prediction API"}


@app.post("/predict", tags=["model"], status_code=200)
def get_prediction(passenger: Passenger):
    model_file = MODEL_DIR / "our_titanic_model.pkl"

    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not found.")

    survived, confidence = predict(passenger=passenger, model_path=model_file)

    return PredictionResult(survived=survived, confidence=confidence)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)
