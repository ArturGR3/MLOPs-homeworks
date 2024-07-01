# metrics.py
from sklearn.metrics import mean_squared_log_error
import numpy as np

def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))