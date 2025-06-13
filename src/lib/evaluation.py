import numpy as np
from sklearn.metrics import r2_score
from .activations import mse_loss


def evaluate_model(
    nn, X_test: np.ndarray, y_test: np.ndarray, y_min: float, y_max: float
) -> tuple[float, float, np.ndarray, np.ndarray]:
    y_pred_norm = nn.forward(X_test)
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    y_actual = y_test * (y_max - y_min) + y_min
    mse = mse_loss(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    return mse, r2, y_actual, y_pred
