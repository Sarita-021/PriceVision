import numpy as np

def smape(y_true, y_pred, eps=1e-6):
    """
    Symmetric Mean Absolute Percentage Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0

    smape_value = np.mean(numerator / (denominator + eps))
    return smape_value * 100
