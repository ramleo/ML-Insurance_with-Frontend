#!/usr/bin/env python3
"""Auto-generated FastAPI prediction API."""
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
    model_config = {"populate_by_name": True}

    id: Optional[float] = None
    Age: Optional[float] = None
    Annual_Income: Optional[float] = Field(None, alias="Annual Income")
    Number_of_Dependents: Optional[float] = Field(None, alias="Number of Dependents")
    Health_Score: Optional[float] = Field(None, alias="Health Score")
    Previous_Claims: Optional[float] = Field(None, alias="Previous Claims")
    Vehicle_Age: Optional[float] = Field(None, alias="Vehicle Age")
    Credit_Score: Optional[float] = Field(None, alias="Credit Score")
    Insurance_Duration: Optional[float] = Field(None, alias="Insurance Duration")
    Gender: Optional[str] = None
    Marital_Status: Optional[str] = Field(None, alias="Marital Status")
    Education_Level: Optional[str] = Field(None, alias="Education Level")
    Occupation: Optional[str] = None
    Location: Optional[str] = None
    Policy_Type: Optional[str] = Field(None, alias="Policy Type")
    Policy_Start_Date: Optional[str] = Field(None, alias="Policy Start Date")
    Customer_Feedback: Optional[str] = Field(None, alias="Customer Feedback")
    Smoking_Status: Optional[str] = Field(None, alias="Smoking Status")
    Exercise_Frequency: Optional[str] = Field(None, alias="Exercise Frequency")
    Property_Type: Optional[str] = Field(None, alias="Property Type")

    def to_pipeline_df(self) -> pd.DataFrame:
        """Return a DataFrame using the original column names the pipeline expects."""
        return pd.DataFrame([{
            "id": self.id,
            "Age": self.Age,
            "Annual Income": self.Annual_Income,
            "Number of Dependents": self.Number_of_Dependents,
            "Health Score": self.Health_Score,
            "Previous Claims": self.Previous_Claims,
            "Vehicle Age": self.Vehicle_Age,
            "Credit Score": self.Credit_Score,
            "Insurance Duration": self.Insurance_Duration,
            "Gender": self.Gender,
            "Marital Status": self.Marital_Status,
            "Education Level": self.Education_Level,
            "Occupation": self.Occupation,
            "Location": self.Location,
            "Policy Type": self.Policy_Type,
            "Policy Start Date": self.Policy_Start_Date,
            "Customer Feedback": self.Customer_Feedback,
            "Smoking Status": self.Smoking_Status,
            "Exercise Frequency": self.Exercise_Frequency,
            "Property Type": self.Property_Type,
        }])

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
    df = data.to_pipeline_df()
    pred = pipeline.predict(df)[0]
    return {"prediction": float(pred)}

@app.post("/predict/batch")
def predict_batch(data: List[InputData]):
    df = pd.concat([d.to_pipeline_df() for d in data], ignore_index=True)
    preds = pipeline.predict(df)
    return {"predictions": preds.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
