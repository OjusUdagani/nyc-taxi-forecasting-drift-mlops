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

# Tell Polars to display all columns and never truncate them with "..."
pl.Config.set_tbl_cols(-1)

from src.ingestion.dataCombine import combineData
from src.ingestion.load import getData

from src.preprocessing.clean import cleanDf
from src.preprocessing.aggregate import agg

from src.preprocessing.splitting import split_df
from src.features.featureEngineering import createFeatures
from src.features.oneHot import oneHotEncode


from src.models.trainPredict import training

from src.evaluations.eval import metrics_gen

from src.drift.covariate_shift_detect import covariate_shift
from src.drift.concept_drift_detect import concept_drift_with_warmup

from src.drift.evidentlyTest import evTest


# Get the directory where run_pipeline.py is located
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# Define robust, system-independent relative paths
readPath = str(PROJECT_ROOT / "data" / "raw" / "*.parquet")
writeMergedPath = str(PROJECT_ROOT / "data" / "processed" / "merged_file.parquet")

print("Loading data... ")
combineData(readPath=readPath, writePath= writeMergedPath)


df = getData(writeMergedPath)

print("Cleaning and Aggregating Data... ")
df = agg(df)
df = cleanDf(df)


print("Training and Evaluating Model using Sliding Window... \n")


start_train_date = datetime(2025, 1, 1)
end_train_date = datetime(2025, 12, 31)

window_size = 4
forecast_size = 1

metrics = []
current_train_start = start_train_date

reports = pd.DataFrame(columns = ["BaseMonths","TestMonths","Report"])


while True:
  current_train_end = current_train_start + relativedelta(months=window_size)
  current_test_end = current_train_end + relativedelta(months=forecast_size)

  if current_test_end > end_train_date:
    break
  
  print("Training months: " + current_train_start.strftime("%b") + "-" + (current_train_end - relativedelta(months=1)).strftime("%b"))
  dfTrain, dfTest = split_df(df, current_train_start, current_train_end, current_test_end)

  print("Feature Engineering... ")
  dfTrain, dfTest= createFeatures(dfTrain, dfTest)

  print("Checking for Drift Detection... ")

  #Data Drift Tests

  #My Covariate Test
  KSTest = covariate_shift(dfTrain, dfTest)

  if KSTest["dataset_drifted"]:
    print(f"There has been a data drift with {KSTest["percentage_features_drifted"]}% of the features thus a covariate shift")
    print("Occured when comparing base months of " + current_train_start.strftime("%B") + "-" + (current_train_end - relativedelta(months=1)).strftime("%B") + " to test month of " + current_train_end.strftime("%B")+ "\n")

    print(f"Detailed results saved to outputs/my_data_drift{current_train_start.strftime("%B") + "-" + (current_train_end - relativedelta(months=1)).strftime("%B")}.csv")


    output_dir = Path("src/outputs")
    output_dir.mkdir(exist_ok=True)

    KSTest["feature_report"].write_csv(output_dir / f"my_data_drift{current_train_start.strftime("%B") + "-" + (current_train_end - relativedelta(months=1)).strftime("%B")}.csv")

    print("/n")

  print("Checking for Drift Detection using Evidently's Library... ")
  #Evidently's Covariate Test
  currentReport = evTest(dfTrain, dfTest, current_train_start, current_train_end)
  reports = pd.concat([reports, currentReport], ignore_index = True)

  

  dfTrain, dfTest, test_timestamps = oneHotEncode(dfTrain, dfTest)
  print("Training... ")
  yTest, yPred = training(dfTrain, dfTest, current_train_start.strftime("%b"), (current_train_end - relativedelta(months=1)).strftime("%b"))

  print("Evaluating on test month: " + current_train_end.strftime("%b") + "\n")
  metric = metrics_gen(yTest, yPred, current_train_start, (current_train_end - relativedelta(months=1)))

  metrics.append(metric)
  

  current_train_start = current_train_start + relativedelta(months=1)


metrics = pl.concat(metrics)

#metrics.json
avg_rmse = metrics['RMSE'].mean()
avg_mae = metrics["MAE"].mean()

print(f"Average RMSE: {avg_rmse}")
print(f"Average MAE: {avg_mae}")

print("Detailed results saved to outputs/backtesting_results.csv")


output_dir = Path("src/outputs")
output_dir.mkdir(exist_ok=True)

metrics.write_csv(output_dir / "backtesting_results.csv")

print("Detailed results of Evidently Drift Test saved to evidentlyResults ")
reports.to_json(output_dir / "evidentlyResults.json", orient="records", indent=4)


