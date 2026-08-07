
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




def evTest(train_df, test_df, current_train_start, current_train_end):
    #Page Hinkley Method
    pHTest = Report([DataDriftPreset()], include_tests= True)
    eval = pHTest.run(current_data = test_df.drop(columns = ['trip_count', 'month', 'year', 'day_of_month','PULocationID']), reference_data = train_df.drop(columns = ['trip_count', 'month', 'year', 'day_of_month', 'PULocationID']))
    return pd.DataFrame([{
        "BaseMonths": current_train_start.strftime("%B") + "-" + (current_train_end - relativedelta(months=1)).strftime("%B"),
        "TestMonths": current_train_end.strftime("%B") ,
        "Report": eval.dict()
    }])

