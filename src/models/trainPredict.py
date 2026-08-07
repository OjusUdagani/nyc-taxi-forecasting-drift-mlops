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


def training(train_df, test_df, startMonth, endMonth):

    #Getting X and y from Train and Test
    X_train, y_train = train_df.drop(columns = 'trip_count'), train_df['trip_count']
    X_test, y_test = test_df.drop(columns = 'trip_count'), test_df['trip_count']


    X_train = X_train.drop(columns=['hour'])
    X_test = X_test.drop(columns=['hour'])


    #Training of Chosen model of XGBoost
    xgModel = xgb.XGBRegressor(
        tree_method = 'hist',
        enable_categorical = True,
        n_estimators=200,     # Maximum number of sequential trees
        max_depth=6,          # Depth of each tree
        learning_rate=0.02,    # Step size shrinkage (eta)

    )

    xgModel.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100)

    y_predX = xgModel.predict(X_test)

    modelName = str(startMonth) + "-" + str(endMonth) + "Model.pkl"

    # Define your target folder and filename
    output_dir = "src/models"
    file_path = os.path.join(output_dir, modelName)

    with open(file_path, "wb") as f:
        pickle.dump(xgModel, f)

    return y_test, y_predX
  



