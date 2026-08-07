import numpy as np

def concept_drift_with_warmup(train_res, test_res, delta=0.005, lambda_thresh=50.0):
    
    # Establish the initial baseline using historical training errors
    abs_train_errors = np.abs(train_res)
    running_mean = np.mean(abs_train_errors)

    # Initialize the sum tracker
    sum_m = 0.0
    min_sum_m = float('inf')

    drift_detected = False
    drift_index = -1

    abs_test_errors = np.abs(test_res)

    # Start checking the test errors with historical context intact
    for t, error in enumerate(abs_test_errors):
        # Update the mean using a continuous sample count
        # (len(train_res) prevents a few early test points from skewing the mean)
        total_samples = len(abs_train_errors) + t + 1
        running_mean = running_mean + (error - running_mean) / total_samples

        # Accumulate the deviations
        sum_m += (error - running_mean - delta)

        if sum_m < min_sum_m:
            min_sum_m = sum_m

        if (sum_m - min_sum_m) > lambda_thresh:
            drift_detected = True
            drift_index = t
            break

    return drift_detected, drift_index