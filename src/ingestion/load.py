import numpy as np
import pandas as pd
import polars as pl


def getData(readPath):
    dataFrame = pl.read_parquet(readPath)
    return dataFrame