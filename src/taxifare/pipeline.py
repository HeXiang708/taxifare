"""Scikit-learn preprocessing pipelines for the taxi-fare features."""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from taxifare import preprocessing as preproc

def build_preproc_pipeline():
    """Build the full preprocessing pipeline for raw ride data.

    Combines the passenger, time and distance sub-pipelines in a
    ``ColumnTransformer``. All steps are stateless (fixed scaling ranges), so
    the pipeline gives identical results at training and inference time.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformer that expects a DataFrame with ``passenger_count``,
        ``pickup_datetime`` and the columns in
        ``preprocessing.lonlat_features``, and outputs 14 features:

        - 1 scaled passenger count
        - 12 time features (7 one-hot day of week, scaled timedelta,
          hour sin/cos, month sin/cos)
        - 1 scaled Manhattan distance

        Columns not listed above are dropped.
    """
    passenger_pipe = make_passenger_pepiline()
    time_pipe = make_time_pepiline()
    distance_pipe = make_dist_pipeline()

    return ColumnTransformer(
        [
            ("passenger_preproc", passenger_pipe, ["passenger_count"]),
            ("time_preproc", time_pipe, ["pickup_datetime"]),
            ("dist_preproc", distance_pipe, preproc.lonlat_features),
        ],
    )

def make_passenger_pepiline():
    """Build the passenger-count preprocessing step.

    Returns
    -------
    sklearn.preprocessing.FunctionTransformer
        Applies :func:`preprocessing.scale_passenger` to ``passenger_count``.
    """
    return FunctionTransformer(preproc.scale_passenger)


def make_time_pepiline():
    """Build the pickup-datetime preprocessing pipeline.

    Extracts time features with :func:`preprocessing.transform_time_features`,
    then one-hot encodes ``day_of_week`` (fixed categories 0-6), scales
    ``timedelta`` with :func:`preprocessing.scale_timedelta`, and passes the
    hour and month sine/cosine features through unchanged.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline that takes ``pickup_datetime`` and outputs 12 columns, in
        order: 7 day-of-week one-hot columns, scaled ``timedelta``, then
        ``hour_sin``, ``hour_cos``, ``month_sin``, ``month_cos``.
    """
    # days of the week from 0 to 6
    days_of_week = np.arange(0, 7, 1)

    day_of_week_encoder = OneHotEncoder(
        categories=[days_of_week],
        handle_unknown="ignore",
        sparse_output=False
    )

    return make_pipeline(
        FunctionTransformer(preproc.transform_time_features),
        make_column_transformer(
            (day_of_week_encoder, ["day_of_week"]),
            (FunctionTransformer(preproc.scale_timedelta), ["timedelta"]),
            remainder="passthrough" # keep hour_sin and hour_cos, and month_sin and month_cos
        )
    )

def make_dist_pipeline():
    """Build the distance preprocessing pipeline.

    Computes the Manhattan distance between pickup and dropoff with
    :func:`preprocessing.manhattan_distance_for_pipe`, then scales it with
    :func:`preprocessing.scale_distance`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline that takes the columns in ``preprocessing.lonlat_features``
        and outputs a single scaled ``distance`` column.
    """
    return make_pipeline(
        FunctionTransformer(preproc.manhattan_distance_for_pipe),
        FunctionTransformer(preproc.scale_distance)
    )
