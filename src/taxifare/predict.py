"""Load a trained taxi-fare model and make fare predictions."""

import joblib
import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas")

from taxifare.geoprocessing import nyc_addr_convert

def load_model(model_path: str):
    """Load a trained model (e.g. a scikit-learn pipeline) from disk.

    Args:
        model_path: Path to the ``.joblib`` file containing the serialized model.

    Returns:
        The deserialized model object.
    """
    return joblib.load(model_path)

def predict(model, data):
    """Predict the taxi fare for a single ride.

    Args:
        model: A fitted model exposing a ``predict`` method, typically the
            pipeline returned by :func:`load_model`.
        data: A ``pandas.DataFrame`` with the raw ride features
            (``pickup_datetime``, pickup/dropoff longitude and latitude,
            ``passenger_count``). Only the first row's prediction is returned.

    Returns:
        The predicted fare for the first row of ``data``.
    """
    if "origin_addr" in data.columns:
        ori_latitude, ori_longitude = nyc_addr_convert(data["origin_addr"].iloc[0])
        dest_latitude, dest_longitude = nyc_addr_convert(data["dest_addr"].iloc[0])
    X_pred=pd.DataFrame([{
        'pickup_datetime': data['pickup_datetime'].iloc[0],
        'pickup_longitude': ori_longitude,
        'pickup_latitude':ori_latitude,
        'dropoff_longitude':dest_longitude,
        'dropoff_latitude':dest_latitude,
        'passenger_count':data['passenger_count'].iloc[0]
    }])
    prediction = model.predict(X_pred)
    return prediction[0]

if __name__ == "__main__":
    X_dummy = pd.DataFrame([{
          'pickup_datetime': '2013-07-06 17:18:00 UTC',
          'pickup_longitude': -73.950655,
          'pickup_latitude': 40.783282,
          'dropoff_longitude': -73.984365,
          'dropoff_latitude': 40.769802,
          'passenger_count': 1
    }])

    X_arr = pd.DataFrame([{
          'pickup_datetime': '2013-07-06 17:18:00 UTC',
          'origin_addr': '201 E 93rd St, New York, NY  10128 United States',
          'dest_addr': '8 Columbus Ave, New York, NY  10023 United States',
          'passenger_count': 1
    }])

    model = load_model("models/linreg_pipeline.joblib")

    print(predict(model, X_arr))
