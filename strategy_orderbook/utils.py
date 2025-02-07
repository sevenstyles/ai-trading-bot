import numpy as np

print("utils.py script starting...")

def calculate_z_score(data):
    """Calculates the Z-score of the last data point."""
    if len(data) < 2:  # Need at least 2 data points for standard deviation
        return 0

    std_dev = np.std(data)
    if std_dev == 0:
        return 0  # Return 0 if standard deviation is zero
    else:
        return (data[-1] - np.mean(data)) / std_dev 