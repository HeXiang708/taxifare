import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from taxifare import preprocessing as preproc

def build_preproc_pipeline():
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
    return FunctionTransformer(preproc.scale_passenger)


def make_time_pepiline():
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
    return make_pipeline(
        FunctionTransformer(preproc.manhattan_distance_for_pipe),
        FunctionTransformer(preproc.scale_distance)
    )
