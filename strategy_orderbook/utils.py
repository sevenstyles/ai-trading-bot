import numpy as np

print("utils.py script starting...")

def calculate_z_score(data_point, data_series):
    """Calculates the Z-score of a single data point against a series."""
    if len(data_series) < 2:  # Need at least 2 data points for standard deviation
        return 0  # Return 0 if not enough data points

    std_dev = np.std(data_series)
    if std_dev == 0:
        std_dev = 0.00001  # Add a small constant to prevent division by zero
    return (data_point - np.mean(data_series)) / std_dev