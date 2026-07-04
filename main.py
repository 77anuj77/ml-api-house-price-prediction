from fastapi import FastAPI, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import Optional, Annotated, Literal
import pandas as pd
import pickle

# Paths
PATH_DB = "/Users/ayushparoha/Documents/Data_Science/env/college/ass9/FAML/Housing.csv"
MODEL_PATH = "house_price_model.pkl"
COLUMNS_PATH = "feature_columns.pkl"

# Load model and feature columns
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(COLUMNS_PATH, "rb") as columns_file:
    feature_columns = pickle.load(columns_file)

# Compute average price per sqft from the dataset (to be used as placeholder for missing price)
def compute_avg_price_per_sqft():
    df = pd.read_csv(PATH_DB)
    return (df["price"] / df["area"]).mean()

AVG_PRICE_PER_SQFT = compute_avg_price_per_sqft()

app = FastAPI()

def load_data():
    try:
        data = pd.read_csv(PATH_DB)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Housing.csv file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save(data):
    try:
        data.to_csv(PATH_DB, index=False)
    except Exception:
        raise HTTPException(status_code=500, detail="Try again")

# Input data model
class DATA(BaseModel):
    area: Annotated[int, Field(..., gt=0, description="Area (in sq ft) of the House")]
    bedrooms: int = Field(..., example=3, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, description="Number of bathrooms")
    stories: int = Field(..., ge=0, description="Number of stories")
    mainroad: Optional[Literal["yes", "no"]] = Field(default="no", description="If house is on the main road")
    guestroom: Optional[Literal["yes", "no"]] = Field(default="no", description="If guest room is present")
    basement: Optional[Literal["yes", "no"]] = Field(default="no", description="If basement is present")
    hotwaterheating: Optional[Literal["yes", "no"]] = Field(default="no", description="If hot water heating is present")
    airconditioning: Optional[Literal["yes", "no"]] = Field(default="no", description="If air conditioning is present")
    parking: Optional[int] = Field(default=0, ge=0, description="Number of parking spaces")
    prefarea: Optional[Literal["yes", "no"]] = Field(default="no", description="If the house is in a preferred area")
    furnishingstatus: Optional[Literal["furnished", "semi-furnished", "unfurnished"]] = Field(default="unfurnished", description="Furnishing status of the house")

@app.get("/")
def about():
    return {"message": "House Price Prediction API is running"}

@app.get("/data")
def data():
    return load_data().to_dict(orient="records")

@app.post("/predict")
def predict(House: DATA):
    # Map yes/no fields to 1/0 for those that were encoded as binary in training
    binary_map = {"yes": 1, "no": 0}
    input_data = {
        "area": House.area,
        "bedrooms": House.bedrooms,
        "bathrooms": House.bathrooms,
        "stories": House.stories,
        "mainroad": binary_map[House.mainroad],
        "guestroom": binary_map[House.guestroom],
        "basement": binary_map[House.basement],
        "airconditioning": binary_map[House.airconditioning],
        "parking": House.parking,
        "prefarea": binary_map[House.prefarea],
        # Keep hotwaterheating as string for one-hot encoding later
        "hotwaterheating": House.hotwaterheating,
        "furnishingstatus": House.furnishingstatus,
        # Placeholder for price per sqft (using average from training data)
        "price_per_sqft": AVG_PRICE_PER_SQFT,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode furnishingstatus (drop first category)
    input_df = pd.get_dummies(input_df, columns=["furnishingstatus"], drop_first=True)

    # Ensure all expected columns are present; fill missing with 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    return {"predicted_price": round(float(prediction), 2)}