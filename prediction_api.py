from pydantic import BaseModel
from fastapi import FastAPI, Query # Note: FastAPI and BaseModel are capitalized
import pickle
import numpy as np
import os
import uvicorn


class DiabetesFeatures(BaseModel):
    # Match the case used in your model training (Pydantic converts to lowercase keys in JSON)
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    


app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes status using pre-trained ML model and scaler. Supports both POST (JSON body) and GET (query parameters).",
    version="1.0"
)


MODEL_PATH = "model/trained_model.sav"
SCALER_PATH = "model/scaler.pkl"

try:
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded successfully from {SCALER_PATH}")

except FileNotFoundError:
    
    raise FileNotFoundError("ERROR: Model files ('trained_model.sav' or 'scaler.pkl') not found. Ensure they are in the 'model/' directory.")
except Exception as e:
    
    raise Exception(f"ERROR during model/scaler loading: {e}")


def get_prediction_result(data_list: list):
    
    if not model or not scaler:
        
        return {"error": "Internal server error: Model or Scaler not loaded."}, 500

    
    input_array = np.asarray(data_list).reshape(1, -1)

    
    scaled_input = scaler.transform(input_array)

    
    prediction = model.predict(scaled_input)[0]

   
    result_text = 'Low Risk of having diabetes (0)' if prediction == 0 else 'High Risk of having diabetes(1)'

    return {
        "Prediction_value": int(prediction),
        "Result": result_text,
        
    }



@app.get("/")
def read_root():
    return {"status": "Diabetes Prediction API is running."}



@app.post("/predict")
def predict_diabetes_post(data: DiabetesFeatures):
   
    input_list = [
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        
    ]
    
    result = get_prediction_result(input_list)
    result["api_method"] = "POST (JSON Body)"
    return result



@app.get("/predict_query")
def predict_diabetes_get(
    
    pregnancies: int = Query(..., description="Number of times pregnant (e.g., 6)", example=6),
    glucose: float = Query(..., description="Plasma glucose concentration (e.g., 148.0)", example=148.0),
    blood_pressure: float = Query(..., description="Diastolic blood pressure (e.g., 72.0)", example=72.0),
    skin_thickness: float = Query(..., description="Triceps skin fold thickness (e.g., 35.0)", example=35.0),
    insulin: float = Query(..., description="2-Hour serum insulin (e.g., 0.0)", example=0.0),
    bmi: float = Query(..., description="Body mass index (e.g., 33.6)", example=33.6),
    diabetes_pedigree_function: float = Query(..., description="Diabetes pedigree function (e.g., 0.627)", example=0.627),
    
):
    
    input_list = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        
    ]

    result = get_prediction_result(input_list)
    
    return result


