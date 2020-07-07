import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def weighted_mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ds = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    maes = ds.groupby('y_true').apply(lambda df: mean_absolute_error(df['y_true'], df['y_pred']))
    return np.mean(maes)
