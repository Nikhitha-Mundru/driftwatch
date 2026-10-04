import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()
model = joblib.load("models/model.joblib")

class PredictInput(BaseModel):
    age: float
    fnlwgt: float
    education_num: float
    capital_gain: float
    capital_loss: float
    hours_per_week: float

@app.post("/predict")
def predict(data: PredictInput):
    start = time.time()
    X = pd.DataFrame([{
        "age": data.age,
        "fnlwgt": data.fnlwgt,
        "education-num": data.education_num,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week
    }])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    latency = time.time() - start
    return {
        "prediction": int(pred),
        "probability_over_50k": float(prob[1]),
        "latency_ms": round(latency * 1000, 2)
    }

@app.get("/health")
def health():
    return {"status": "ok"}