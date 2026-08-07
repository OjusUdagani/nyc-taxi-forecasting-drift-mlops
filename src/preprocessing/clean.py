import numpy as np
import math
import pandas as pd
import glob
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns



def cleanDf(df):
    df = df.filter(pl.col('year') >= 2025)
    return df