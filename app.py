import pickle
import pandas as pd
from sklearn import preprocessing
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware as CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the API!"}

# Define endpoint for making predictions
@app.post('/predict-head-direction')
def predict(data:dict):
  # Load model from .pkl file
  with open('./headdirection_model.pkl','rb') as file:
    model = pickle.load(file)
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Make prediction
    prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}
    #df[['wrapping','depth','adultsubadult','facebundles','preservation','length']]

@app.post('/predict-wrapping')
def predict(data:dict):
  # Load model from .pkl file
  with open('./wrapping_model.pkl','rb') as file:
    model = pickle.load(file)
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Make prediction
    prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}