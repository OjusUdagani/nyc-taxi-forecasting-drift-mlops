import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import polars as pl


def covariate_shift(df_base, df_test):

    com_numeric_col = (df_base.select_dtypes(include = [np.number])
    .columns.intersection(df_test.select_dtypes(include = [np.number]).columns)
    )
    # Explicitly drop target, dates, and ID features
    colIgnore = ['trip_count', 'month', 'year', 'day_of_month', 'PULocationID']

    features = []
    ks_distances = []
    p_values = []
    feature_drifted = []

    # Set a practical distance threshold instead of relying on p-value
    # 0.05 to 0.10 is standard for large-sample drift detection
    DISTANCE_THRESHOLD = 0.05

    for col in com_numeric_col:
        if col not in colIgnore:
            sample1 = df_base[col].dropna()
            sample2 = df_test[col].dropna()

            if len(sample1) == 0 or len(sample2) == 0:
                continue

            # Perform K-S Test
            stat, p = ks_2samp(sample1, sample2, method="asymp")

            features.append(col)
            ks_distances.append(stat)
            p_values.append(p)

            # Flag drift based on distance, NOT p-value
            feature_drifted.append(stat > DISTANCE_THRESHOLD)

    report = pl.DataFrame({
        "Feature": features,
        "K-S Distance (D)": ks_distances,
        "Raw p-value": p_values,
        "Feature Drifted": feature_drifted
    })

    drifted_features_count = report['Feature Drifted'].sum()

    # Guard against division by zero if no valid features remain
    pct_drifted = drifted_features_count / len(features) if features else 0.0

    # Dataset is considered drifted if more than 20% of your operational features genuinely drift
    dataset_drifted = pct_drifted > 0.20

    return {
        "dataset_drifted": dataset_drifted,
        "percentage_features_drifted": round(pct_drifted * 100, 2),
        "feature_report": report
    }