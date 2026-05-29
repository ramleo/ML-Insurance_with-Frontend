#!/usr/bin/env python3
"""Auto-generated FastAPI prediction API."""
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import joblib, pandas as pd

app = FastAPI(title="ML Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = joblib.load("models/final_pipeline.pkl")

class InputData(BaseModel):
    id: Optional[float] = None
    Age: Optional[float] = None
    Annual Income: Optional[float] = None
    Number of Dependents: Optional[float] = None
    Health Score: Optional[float] = None
    Previous Claims: Optional[float] = None
    Vehicle Age: Optional[float] = None
    Credit Score: Optional[float] = None
    Insurance Duration: Optional[float] = None
    Gender: Optional[str] = None
    Marital Status: Optional[str] = None
    Education Level: Optional[str] = None
    Occupation: Optional[str] = None
    Location: Optional[str] = None
    Policy Type: Optional[str] = None
    Policy Start Date: Optional[str] = None
    Customer Feedback: Optional[str] = None
    Smoking Status: Optional[str] = None
    Exercise Frequency: Optional[str] = None
    Property Type: Optional[str] = None

@app.get("/")
def index():
    """Serve the prediction UI if index.html exists."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "ML Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = pipeline.predict(df)[0]
    return {"prediction": float(pred)}

@app.post("/predict/batch")
def predict_batch(data: List[InputData]):
    df = pd.DataFrame([d.dict() for d in data])
    preds = pipeline.predict(df)
    return {"predictions": preds.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
