from fastapi import FastAPI
import joblib
import pandas as pd
from taxifare.predict import predict
# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    """Root endpoint returning a greeting."""
    return {"greeting": "Hello !"}


# Prediction endpoint
@app.get("/predict")
def prediction(
    pickup_datetime:float,
    pickup_longitude:float,
    pickup_latitude:float,
    dropoff_longitude:float,
    dropoff_latitude:float,
    passenger_count: int
):
    X_pred=pd.DataFrame([{
        'pickup_datetime': pickup_datetime,
        'pickup_longitude':pickup_longitude,
        'pickup_latitude':pickup_latitude,
        'dropoff_longitude':dropoff_longitude,
        'dropoff_latitude':dropoff_latitude,
        'passenger_count':passenger_count
    }])

    model=joblib.load('models/linreg_pipeline.joblib')
    y_pred,y_proba=predict(model, X_pred)

    """Prediction endpoint that returns the fare amount."""
    return {"Prediction": y_pred[0],'Probability':y_proba[0]}



# Dummy endpoint
@app.get("/dummy")
def dummy(number: int,):
    """Dummy endpoint that returns the square of the input number."""
    return {"result": number ** 2}
