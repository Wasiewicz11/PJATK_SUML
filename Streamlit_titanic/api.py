import uvicorn
from fastapi import FastAPI

from routers import model

app = FastAPI()
app.include_router(model.router)


@app.get("/", tags=["intro"])
def index():
    return {"message": "Titanic Survival Prediction API"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)
