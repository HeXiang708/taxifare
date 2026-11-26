import math
from typing import TypeVar

import numpy as np
import pandas as pd

def scale_passenger(p):
    """Min-max scale the passenger count to the [0, 1] range.

    Uses a fixed range of 0 to 8 passengers instead of statistics fitted on
    the training data, so the transformation is stateless and identical at
    training and inference time.

    Parameters
    ----------
    p : array-like, pd.Series or pd.DataFrame
        Passenger count(s).

    Returns
    -------
    Same type as ``p``
        Scaled passenger count. Values above 8 are scaled above 1.0 (not clipped).
    """
    p_min = 0.
    p_max = 8.
    p_scaled = (p - p_min) / (p_max - p_min)
    return p_scaled

def transform_time_features(X: pd.DataFrame):
    """Extract time-based features from the pickup datetime.

    The datetime is converted to New York local time before extracting the
    hour, day of week and month. Hour and month are cyclically encoded with
    sine/cosine so that e.g. 23h and 0h end up close to each other.

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Must contain a ``pickup_datetime`` column (or be a Series named
        ``pickup_datetime``). Naive datetimes are assumed to be UTC.

    Returns
    -------
    pd.DataFrame
        Same index as ``X``, with columns:

        - ``hour_sin``, ``hour_cos``: cyclical encoding of the local hour (0-23)
        - ``day_of_week``: local day of week (0 = Monday, 6 = Sunday)
        - ``month_sin``, ``month_cos``: cyclical encoding of the month (1-12)
        - ``timedelta``: days elapsed since 2009-01-01 00:00 UTC (float)
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()

    pickup_dt = pd.to_datetime(X["pickup_datetime"], utc=True)
    timedelta = (pickup_dt - pd.Timestamp('2009-01-01T00:00:00', tz='UTC')) / pd.Timedelta(1, 'D')

    pickup_dt_ny = pickup_dt.dt.tz_convert("America/New_York").dt

    dow = pickup_dt_ny.weekday
    hour = pickup_dt_ny.hour
    month = pickup_dt_ny.month

    hour_sin = np.sin(2 * math.pi / 24 * hour)
    hour_cos = np.cos(2 * math.pi / 24 * hour)

    month_sin = np.sin(2 * math.pi / 12 * month)
    month_cos = np.cos(2 * math.pi / 12 * month)

    return pd.DataFrame({
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_of_week': dow,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'timedelta': timedelta
    }, index=X.index)

def scale_timedelta(timedelta):
    """Min-max scale the ``timedelta`` feature to roughly the [0, 1] range.

    Uses a fixed range of 0 to 2190 days (about 6 years, 2009-01-01 to
    2015-01-01), matching the date range of the training data.

    Parameters
    ----------
    timedelta : array-like, pd.Series or pd.DataFrame
        Days elapsed since 2009-01-01, as returned by
        :func:`transform_time_features`.

    Returns
    -------
    Same type as ``timedelta``
        Scaled values. Dates after 2015-01-01 are scaled above 1.0 (not clipped).
    """
    timedelta_min = 0
    timedelta_max = 2190 # Our model may extend in the future. No big deal if the scaled data extend slightly beyond 1.0

    scaled = (timedelta - timedelta_min) / (timedelta_max - timedelta_min)
    return scaled

def manhattan_distance_vectorized(df: pd.DataFrame, start_lat: str, start_lon: str, end_lat: str, end_lon: str):
    """Compute the Manhattan distance in km between two points, row by row.

    The distance is ``(|dlat| + |dlon|) * earth_radius``, with the
    differences in radians. Longitude differences are not scaled by
    ``cos(latitude)``, so east-west distances are overestimated (about 32%
    at New York's latitude).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the coordinate columns, in decimal degrees.
    start_lat, start_lon : str
        Column names of the start point latitude and longitude.
    end_lat, end_lon : str
        Column names of the end point latitude and longitude.

    Returns
    -------
    pd.Series
        Manhattan distance in km, with the same index as ``df``.
    """
    earth_radius = 6371

    lat_1_rad, lon_1_rad = np.radians(df[start_lat]), np.radians(df[start_lon])
    lat_2_rad, lon_2_rad = np.radians(df[end_lat]), np.radians(df[end_lon])

    dlon_rad = lon_2_rad - lon_1_rad
    dlat_rad = lat_2_rad - lat_1_rad

    manhattan_rad = np.abs(dlon_rad) + np.abs(dlat_rad)
    manhattan_km = manhattan_rad * earth_radius

    return manhattan_km

lonlat_features = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

def manhattan_distance_for_pipe(df: pd.DataFrame):
    """Compute the pickup-to-dropoff Manhattan distance, for use in a pipeline.

    Wraps :func:`manhattan_distance_vectorized` with the column names in
    ``lonlat_features``. The order of ``lonlat_features`` must match the
    ``(start_lat, start_lon, end_lat, end_lon)`` signature.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in ``lonlat_features``.

    Returns
    -------
    pd.DataFrame
        A single ``distance`` column (km), with the same index as ``df``.
    """
    distance = manhattan_distance_vectorized(df, *lonlat_features)
    return pd.DataFrame({'distance': distance})

def scale_distance(dist):
    """Min-max scale the distance to roughly the [0, 1] range.

    Uses a fixed range of 0 to 100 km.

    Parameters
    ----------
    dist : array-like, pd.Series or pd.DataFrame
        Distance(s) in km, as returned by :func:`manhattan_distance_for_pipe`.

    Returns
    -------
    Same type as ``dist``
        Scaled distance. Distances above 100 km are scaled above 1.0 (not clipped).
    """
    dist_min = 0
    dist_max = 100
    scaled = (dist - dist_min) / (dist_max - dist_min)
    return scaled
