from pathlib import Path
import joblib

import pandas as pd
from fastapi import FastAPI, HTTPException

from backend.schemas import CreditRiskInput

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "stacking_classifier_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict whether a credit applicant is good or bad risk.",
    version="1.0.0",
)

# Keep this order identical to the columns used to fit the scaler.
FEATURE_COLUMNS = [
    "Age",
    "Job",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Sex_male",
    "Housing_own",
    "Housing_rent",
    "Purpose_car",
    "Purpose_domestic appliances",
    "Purpose_education",
    "Purpose_furniture/equipment",
    "Purpose_radio/TV",
    "Purpose_repairs",
    "Purpose_vacation/others",
]

# replacing the mappings with int data type
SAVING_ACCOUNT_MAP = {
    "No Account": 0,
    "little": 1,
    "moderate": 2,
    "quite rich": 3,
    "rich": 4,
}

CHECKING_ACCOUNT_MAP = {
    "No Account": 0,
    "little": 1,
    "moderate": 2,
    "rich": 3,
}

# function to preprocess input data
def preprocess_input(data : CreditRiskInput) -> pd.DataFrame:
    row = {
        "Age" : data.Age,
        "Job" : data.Job,
        "Saving accounts" : SAVING_ACCOUNT_MAP[
            data.Saving_account
        ],
        
        "Checking account" : CHECKING_ACCOUNT_MAP[
            data.Checking_account
        ],
        
        "Credit Amount" : data.Credit_amount,
        "Duration" : data.Duration,
        
        # One hot encoding
        "Sex_male" : int(data.Sex == "male"),
        "Housing_own" : int(data.Housing == "own"),
        "Housing_rent" : int(data.Housing == "rent"),
        "Purpose_car" : int(data.Purpose == "car"),
        "Purpose_domestic appliances": int(
            data.Purpose == "domestic appliances"
        ),
        "Purpose_education": int(data.Purpose == "education"),
        "Purpose_furniture/equipment": int(
            data.Purpose == "furniture/equipment"
        ),
        "Purpose_radio/TV": int(data.Purpose == "radio/TV"),
        "Purpose_repairs": int(data.Purpose == "repairs"),
        "Purpose_vacation/others": int(
            data.Purpose == "vacation/others"
        ),
    }
    
    # explicitly enforce the feature names and order
    df = pd.DataFrame([row])  # [row] treats dictionary data as one row of data
    return df.reindex(
        columns= FEATURE_COLUMNS,
        fill_value= 0
    )
    
@app.get("/")
def home():
    return {
        "message" : "Credit Risk Prediction API is running"
    }
    
@app.post("/predict")
def predict(data : CreditRiskInput):
    try:
        input_df = preprocess_input(data)
        
        scaled_input = scaler.transform(input_df)
        
        prediction = model.predict(scaled_input)[0]
        
        risk_mapping = {
            0 : "Good",
            1 : "Bad"
        }
        
        predicted_risk = risk_mapping.get(
            int(prediction)
        )
        
        response = {
            "prediction" : int(prediction),
            "risk"  : predicted_risk
        }
        
        return response
    
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code= 400,
            detail = f"Could not process input : {e}"
        ) from e
    
    
    
    
    
    
    
    
    