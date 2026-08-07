import numpy as np
import math
import pandas as pd
import polars as pl
from datetime import date

def split_df(df, start_train_date, end_train_date, end_test_date):

    #Creating location x Hour column 
    df = df.with_columns(
    (pl.col('PULocationID').cast(str) + '_' + pl.col('hour').cast(str)).alias('loc_hour_key')
)

    df = df.sort(['PULocationID', 'tpep_pickup_datetime'])

    # Creating the 1-hour and 2-hour lag features using whole dataset

    df = df.with_columns([
        pl.col('trip_count').shift(1).over('PULocationID').alias('trip_count_lag_1h'),
        pl.col('trip_count').shift(2).over('PULocationID').alias('trip_count_lag_2h')
    ])

    train_df = df.filter(
      (pl.col("tpep_pickup_datetime") >= start_train_date) &
      (pl.col("tpep_pickup_datetime") < end_train_date)
  )

    test_df = df.filter(
        (pl.col("tpep_pickup_datetime") >= end_train_date) &
        (pl.col("tpep_pickup_datetime") < end_test_date)
    )

    return train_df, test_df

