import numpy as np
import math
import pandas as pd
import polars as pl
from datetime import date


def createFeatures(train_df, test_df):
    
    

    loc_hour_avg = (
    train_df.group_by('loc_hour_key')
    .agg(pl.col('trip_count').mean().alias('loc_hour_avg'))
  )
    global_avg = train_df['trip_count'].mean()


    train_df = train_df.join(loc_hour_avg, on='loc_hour_key', how='left')
    test_df = test_df.join(loc_hour_avg, on='loc_hour_key', how='left')
    test_df = test_df.with_columns(pl.col('loc_hour_avg').fill_null(global_avg))

    train_df = train_df.with_columns([
        pl.col('trip_count_lag_1h').fill_null(pl.col('loc_hour_avg')),
        pl.col('trip_count_lag_2h').fill_null(pl.col('loc_hour_avg'))
    ])

    test_df = test_df.with_columns([
        pl.col('trip_count_lag_1h').fill_null(pl.col('loc_hour_avg')),
        pl.col('trip_count_lag_2h').fill_null(pl.col('loc_hour_avg'))
    ])


    train_df = train_df.drop(['loc_hour_key'])
    test_df = test_df.drop(['loc_hour_key'])



    train_df = train_df.to_pandas()
    test_df = test_df.to_pandas()


    return train_df, test_df