import numpy as np
from typing import List

def report_feature_weights(weights: np.ndarray, feature_names: List[str]) -> None:
    print("\nPesos efetivos por atributo:")
    for name, w in zip(feature_names, weights):
        print(f"{name:<20s}: {w:+.6f}")
    print()