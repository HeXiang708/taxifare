from fastapi import FastAPI
import pandas as pd
from taxifare.predict import load_model, predict
# FastAPI instance
app = FastAPI()
#model
model=load_model('models/linreg_pipeline.joblib')

# Root endpoint
@app.get("/")
def root():
    """Root endpoint returning a greeting."""
    return {"greeting": "Hello !"}


# Prediction endpoint
@app.get("/predict")
def prediction(
    pickup_datetime:str,
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

    y_pred=predict(model, X_pred)

    """Prediction endpoint that returns taxi fare amount."""
    return {"fare": y_pred}



# Dummy endpoint
@app.get("/dummy")
def dummy(number: int,):
    """Dummy endpoint that returns the square of the input number."""
    return {"result": number ** 2}
