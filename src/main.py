# Put the code for your API here.
import pandas as pd
import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference


# Load the pickle files
with open('./model/rfc_model.pkl', 'rb') as pickle_file:
    rfc_model = pickle.load(pickle_file)

with open('./model/encoder.pkl', 'rb') as pickle_file:
    encoder = pickle.load(pickle_file)

with open('./model/lb.pkl', 'rb') as pickle_file:
    lb = pickle.load(pickle_file)


class UserRequest(BaseModel):
    age: int = Field(None, example=39)
    workclass: str = Field(None, example="State-gov")
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example="Bachelors") 
    educationNum: int = Field(None, alias="education-num", example=13)
    maritalStatus: str = Field(None, alias="marital-status", example="Never-married")
    occupation: str = Field(None, example="Adm-clerical")
    relationship: str = Field(None, example="Not-in-family")
    race: str = Field(None, example="White") 
    sex: str = Field(None, example="Female") 
    capitalGain: int = Field(None, alias="capital-gain", example=2174)
    capitalLoss: int = Field(None, alias="capital-loss", example=0)
    hoursPerWeek: int = Field(None, alias="hours-per-week", example=40)
    nativeCountry: str = Field(None, alias="native-country", example="United-States")
    
    class Config:
        # utilizes pydantic to replace the name with the alias
        allow_population_by_field_name = True


app = FastAPI()


@app.get("/")
async def welcome_message():
    return {"greeting": "Hello to Udacity ML DevOps - Project 3"}


@app.post("/predict")
async def model_predict(body: UserRequest):
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    # Create model input
    model_input_df = pd.DataFrame(body.dict(by_alias=True), index=[0])

    # Proces the data and call the model 
    model_input_df_processed, _, _, _ = process_data(
    model_input_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)
    prediction = inference(model=rfc_model, X=model_input_df_processed)

    # Format and return result 
    return {"Salary": "Less than or equal to $50k" if prediction[0] == 0 else "Higher than $50k"}
