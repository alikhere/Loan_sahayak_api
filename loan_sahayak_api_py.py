# Import necessary modules and classes
from fastapi import FastAPI, HTTPException  # FastAPI framework and exception handling
from fastapi.middleware.cors import CORSMiddleware  # CORS middleware for handling cross-origin requests
from pydantic import BaseModel  # BaseModel for request data validation
import pickle  # For loading the saved ML model
import pandas as pd  # Pandas library for data manipulation
import joblib  # For loading the saved scaler

# Load the trained machine learning model
model = pickle.load(open('loan_status_predict.sav', 'rb'))

# Define the columns that need to be scaled
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Load the scaler object for feature scaling
scaler = joblib.load('vector.pkl')

# Initialize the FastAPI application
app = FastAPI()

# Define allowed origins for cross-origin resource sharing (CORS)
origins = [
    "http://localhost:5173",  # Local development server (React)
    "https://loan-sahayak-z5n7.onrender.com",  # Deployed frontend application
]

# Add middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow sending credentials (cookies, etc.)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the schema for incoming prediction requests using Pydantic, Ensures that incoming data conforms to a specific structure and type. 
class PredictionRequest(BaseModel):
    Gender: int  # Gender of the applicant (e.g., 1 for Male, 0 for Female)
    Married: int  # Marital status (e.g., 1 for Married, 0 for Not Married)
    Dependents: int  # Number of dependents
    Education: int  # Education level (e.g., 1 for Graduate, 0 for Not Graduate)
    Self_Employed: int  # Employment status (e.g., 1 for Self-Employed, 0 for Not Self-Employed)
    ApplicantIncome: int  # Monthly income of the applicant
    CoapplicantIncome: float  # Monthly income of the co-applicant
    LoanAmount: float  # Loan amount requested
    Loan_Amount_Term: float  # Loan repayment term in months
    Credit_History: float  # Credit history (e.g., 1 for good credit, 0 for no/poor credit)
    Property_Area: int  # Property area type (e.g., 1 for Urban, 2 for Rural, etc.)

# Define an endpoint for loan prediction
@app.post('/predict')
def predict(request: PredictionRequest):
    # Convert the incoming request data into a Pandas DataFrame
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

    # Scale the required columns using the preloaded scaler
    input_df[cols] = scaler.transform(input_df[cols])

    try:
        # Predict the loan status using the ML model
        prediction = model.predict(input_df)

        # Return the prediction as a response
        return {'prediction': int(prediction[0])}
    except Exception as e:
        # Raise an HTTP exception if prediction fails
        raise HTTPException(status_code=400, detail=str(e))

# Run the application when the script is executed directly
if __name__ == '__main__':
    import uvicorn  # Import uvicorn for running the server
    uvicorn.run(app, host='0.0.0.0', port=8028)  # Start the FastAPI server
