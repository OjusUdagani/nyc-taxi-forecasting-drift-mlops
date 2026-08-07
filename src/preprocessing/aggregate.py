import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def agg(df):
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    aggregated_df = (
        df
        # 1. Truncate the datetime column directly to the hour mark
        .with_columns(
            pl.col('tpep_pickup_datetime').dt.truncate('1h')
        )
        # 2. Group by the truncated hour and zone
        .group_by(['tpep_pickup_datetime', 'PULocationID'])
        .agg([
            pl.len().alias('trip_count')


        ])
        # 3. Create the descriptive columns from the grouped timestamp
        .with_columns([
            pl.col('tpep_pickup_datetime').dt.hour().alias('hour'),
            pl.col('tpep_pickup_datetime').dt.strftime('%A').alias('day_of_week'),
            pl.col('tpep_pickup_datetime').dt.day().alias('day_of_month'),
            pl.col('tpep_pickup_datetime').dt.year().alias('year'),
            pl.col('tpep_pickup_datetime').dt.month().alias('month'),
            pl.col('tpep_pickup_datetime').dt.weekday().is_in([6, 7]).alias('is_weekend')
        ])
    )

    aggregated_df = aggregated_df.with_columns(
        pl.col("day_of_week")
        .cast(pl.Enum(weekday_order))
        .alias("day_factor")
    )
    aggregated_df = aggregated_df.drop("day_of_week")

    aggregated_df = aggregated_df.sort("tpep_pickup_datetime")

    return aggregated_df