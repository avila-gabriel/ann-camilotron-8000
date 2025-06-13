import matplotlib.pyplot as plt
import numpy as np

def plot_results(y_actual: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_actual, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r', linewidth=2)
    plt.xlabel('Real')
    plt.ylabel('Predito')
    plt.title('Real vs Predito')
    plt.tight_layout()
    plt.show()