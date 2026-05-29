import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path

from models import Passenger, PredictionResult, TrainConfiguration
from libs.model import predict, train

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("ml_models")
DATA_DIR = Path(BASE_DIR).joinpath("data")

app = FastAPI()


@app.get("/", tags=["intro"])
def index():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/model/train", tags=["model"], status_code=200)
async def train_model(configuration: TrainConfiguration):
    train_data_file = DATA_DIR / "train.csv"
    test_data_file = DATA_DIR / "test.csv"
    labels_file = DATA_DIR / "gender_submission.csv"
    
    model_file = Path(MODEL_DIR).joinpath("our_titanic_model.pkl")

    n_estimators = configuration.n_estimators or 300
    max_depth = configuration.max_depth

    metrics = train(
        n_estimators=n_estimators,
        max_depth=max_depth,
        path2train=train_data_file,
        path2test=test_data_file,
        path2labels=labels_file,
        path2pickle=model_file
    )

    return {"model_fit": "OK", "model_save": "OK", **metrics}

@app.post("/predict", tags=["model"], status_code=200)
def get_prediction(passenger: Passenger):
    model_file = MODEL_DIR / "our_titanic_model.pkl"

    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not found.")

    survived, confidence = predict(passenger=passenger, model_path=model_file)

    return PredictionResult(survived=survived, confidence=confidence)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)
