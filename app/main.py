from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import logging
from datetime import datetime

# 1. Initialize FastAPI
app = FastAPI(title="Motor Maintenance Prediction API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load the trained model
# Using a relative path from the root directory
MODEL_PATH = os.path.join("models", "motor_model.pkl")
model = joblib.load(MODEL_PATH)

# 3. Define the Input Data Schema
class MotorFeatures(BaseModel):
    voltage_v: float = Field(..., ge=180, le=280, description="Motor voltage in volts")
    current_a: float = Field(..., ge=0, le=30, description="Current draw in amperes")
    temp_c: float = Field(..., ge=-10, le=120, description="Temperature in Celsius")
    vibration_g: float = Field(..., ge=0, le=1, description="Vibration level")
    
    class Config:
        schema_extra = {
            "example": {
                "voltage_v": 230,
                "current_a": 12,
                "temp_c": 65,
                "vibration_g": 0.08
            }
        }

# 4. Endpoints
@app.get("/")
def read_root():
    return {"status": "API is Online", "model": "Random Forest v1"}

@app.post("/predict")
def predict_failure(data: MotorFeatures):
    # Convert incoming JSON to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Feature engineering
    input_df = input_df.copy()
    input_df['voltage_deviation'] = (input_df['voltage_v'] - 230).abs()
    input_df['power_usage'] = input_df['voltage_v'] * input_df['current_a']
    input_df['thermal_stress'] = input_df['temp_c'] / 100
    
    # Get prediction (0 or 1)
    prediction = model.predict(input_df)[0]
    
    # Get probability (how sure is the model?)
    probability = model.predict_proba(input_df).max()
    
    # Log the prediction
    logger.info(f"Prediction: {prediction}, Confidence: {probability}, Input: {data.dict()}")
    
    return {
        "failure_prediction": int(prediction),
        "status": "FAILURE RISK" if prediction == 1 else "NORMAL",
        "confidence": round(float(probability), 4),
        "timestamp": datetime.utcnow().isoformat()
    }
