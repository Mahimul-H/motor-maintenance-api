from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Initialize FastAPI
app = FastAPI(title="Motor Maintenance Prediction API")

# 2. Load the trained model
# Using a relative path from the root directory
MODEL_PATH = os.path.join("models", "motor_model.pkl")
model = joblib.load(MODEL_PATH)

# 3. Define the Input Data Schema
class MotorFeatures(BaseModel):
    voltage_v: float
    current_a: float
    temp_c: float
    vibration_g: float

# 4. Endpoints
@app.get("/")
def read_root():
    return {"status": "API is Online", "model": "Random Forest v1"}

@app.post("/predict")
def predict_failure(data: MotorFeatures):
    # Convert incoming JSON to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Get prediction (0 or 1)
    prediction = model.predict(input_df)[0]
    
    # Get probability (how sure is the model?)
    probability = model.predict_proba(input_df).max()
    
    return {
        "failure_prediction": int(prediction),
        "status": "🚨 FAILURE RISK" if prediction == 1 else "✅ NORMAL",
        "confidence": round(float(probability), 4)
    }
