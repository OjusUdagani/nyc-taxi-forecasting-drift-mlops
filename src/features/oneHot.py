import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import pathlib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dateutil.relativedelta import relativedelta
import xgboost as xgb
from evidently import Report
from evidently.presets import DataDriftPreset
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
import json
from pathlib import Path



def oneHotEncode(train_df, test_df):

    train_df = pl.from_pandas(train_df)
    test_df = pl.from_pandas(test_df)
    #Categorical Columns
    Catcolumns = ['day_factor', 'PULocationID']
     # Step 2: recombine to one-hot encode consistently, then re-split
    combined = pl.concat([train_df.with_columns(pl.lit('train').alias('_split')),
                        test_df.with_columns(pl.lit('test').alias('_split'))])


    combined = combined.to_dummies(Catcolumns, drop_first=True)

    combined = combined.with_columns([
        (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos')
    ])

    test_timestamps = test_df['tpep_pickup_datetime'].to_numpy()


    combined = combined.drop('tpep_pickup_datetime')
    train_df = combined.filter(pl.col('_split') == 'train').drop('_split')
    test_df = combined.filter(pl.col('_split') == 'test').drop('_split')


    train_df = train_df.to_pandas()
    test_df = test_df.to_pandas()

    return train_df, test_df, test_timestamps



