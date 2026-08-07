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
import pickle
import os



def metrics_gen(y_test, y_predX, start, end):

    res = np.abs(y_predX - y_test)
    mse = mean_squared_error(y_true = y_test, y_pred = y_predX)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_predX)
    r2 = r2_score(y_test, y_predX)
    
    window_results = pl.DataFrame({
        "Train Months": (start.strftime("%b") )+ "-" + (end.strftime("%b")),
        "Test Month": (end + relativedelta(months=1)).strftime("%b"),
        "MSE" : mse,
        "RMSE": rmse,
        "MAE": mae,
        "R^2": r2
        })
    return window_results
    

    