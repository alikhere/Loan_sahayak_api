# -*- coding: utf-8 -*-
"""
"""

from fastapi import FastAPI, HTTPException #FastAPI: This is the main class used to create a FastAPI application.
#HTTPException: This is used to raise HTTP exceptions, such as returning an error if something goes wrong during the prediction process.
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel #This is a base class from Pydantic used for data validation. It ensures that incoming request data matches the expected structure.
import pickle   #This module is used to load the trained machine learning model from a file.

# Load the trained model

model = pickle.load(open('loan_status_predict.sav', 'rb'))
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'] 
from sklearn.preprocessing import StandardScaler 
st = StandardScaler()


# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost:5173",  # React dev server
    "https://grih-khoj.onrender.com",  # Your deployed frontend
]

app.add_middleware(     #add_middleware(): This function adds the CORSMiddleware to the FastAPI application, allowing specified origins to access the API.
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True, #Allows cookies to be included in cross-origin requests.
    allow_methods=["*"],    #Allows all HTTP methods (GET, POST, PUT, DELETE, etc.) to be used in requests.
    allow_headers=["*"],
)

# Define the request body using Pydantic
class PredictionRequest(BaseModel): #This class defines the structure of the incoming request data using Pydantic's BaseModel
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

@app.post('/predict')   #This decorator defines a POST endpoint at /predict. When a POST request is made to this endpoint, the predict function is executed.
def predict(request: PredictionRequest):    #The predict function takes a PredictionRequest object as input, which contains the validated request data.
    # Extract features from the request data
    features = [
        request.Gender,
        request.Married,
        request.Dependents,
        request.Education,
        request.Self_Employed,
        request.ApplicantIncome,
        request.CoapplicantIncome,
        request.LoanAmount,
        request.Loan_Amount_Term,
        request.Credit_History,
        request.Property_Area
    ]
    #Above block of code extracts the feature values from the incoming request and stores them in a list called features. This list will be used as input to the machine learning model.
    # Make prediction
    features[cols] = st.fit_transform(features[cols])
    try:
        prediction = model.predict([features])  #The features list is wrapped in another list because predict expects a 2D array.
        return {'prediction': int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


#responsible for starting your FastAPI application when the script is run directly.
if __name__ == '__main__':  #This pattern is common in Python scripts to ensure that certain code (e.g., starting a web server, running tests) only runs when the script is executed directly and not when it's imported as a module.


    import uvicorn  #Uvicorn is an ASGI (Asynchronous Server Gateway Interface) server that is used to run FastAPI applications.
    uvicorn.run(app, host='0.0.0.0', port=8004)
    #'0.0.0.0' means that the application will be accessible on all available network interfaces of the machine running the script. This makes the API accessible not just from localhost, but also from any external devices that can reach the server over the network.

    
    



