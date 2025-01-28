from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import joblib

# Load the trained ML model and scaler
model = pickle.load(open('loan_status_predict.sav', 'rb'))
scaler = joblib.load('vector.pkl')

# Columns to be scaled
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Initialize the FastAPI application
app = FastAPI()

# CORS settings
origins = [
    "http://localhost:5173",
    "https://loan-sahayak-z5n7.onrender.com",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request schema
class PredictionRequest(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: int

# Define the prediction endpoint
@app.post('/predict')
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([{
        'Gender': request.Gender,
        'Married': request.Married,
        'Dependents': request.Dependents,
        'Education': request.Education,
        'Self_Employed': request.Self_Employed,
        'ApplicantIncome': request.ApplicantIncome,
        'CoapplicantIncome': request.CoapplicantIncome,
        'LoanAmount': request.LoanAmount,
        'Loan_Amount_Term': request.Loan_Amount_Term,
        'Credit_History': request.Credit_History,
        'Property_Area': request.Property_Area
    }])

    # Scale the required columns
    input_df[cols] = scaler.transform(input_df[cols])

    try:
        prediction = model.predict(input_df)
        return {'prediction': int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8028)
