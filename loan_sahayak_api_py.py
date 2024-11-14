from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd  # Import pandas
from sklearn.preprocessing import StandardScaler 
st = StandardScaler()

# Load the trained model
model = pickle.load(open('loan_status_predict.sav', 'rb'))

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost:5173",  # React dev server
    "https://loan-sahayak-z5n7.onrender.com/",  # Your deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body using Pydantic
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

@app.post('/predict')
def predict(request: PredictionRequest):
    # Convert input data to DataFrame with column names
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
    # Make prediction
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'] 
    input_df[cols] = st.fit_transform(input_df[cols])

    try:
        prediction = model.predict(input_df)
        return {'prediction': int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8021)
