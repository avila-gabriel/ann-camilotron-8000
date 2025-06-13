import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split

#### Remoção de outliers (IQR) ####
def remove_outliers_iqr(df: pd.DataFrame, factor: float = 1.5, id_col: str = "ID") -> pd.DataFrame:
    """
    Remove linhas contendo outliers em qualquer coluna numérica (exceto id_col).
    """
    num_cols = [c for c in df.select_dtypes(exclude="object").columns if c != id_col]
    mask = pd.Series(False, index=df.index)
    for col in num_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        mask |= (df[col] < lower) | (df[col] > upper)
    if mask.any():
        removed_ids = df.loc[mask, id_col].tolist()
        print(f"Outliers removidos (IDs): {removed_ids}")
    return df.loc[~mask].reset_index(drop=True)

#### Carregamento e pré-processamento ####
def load_and_preprocess(
    filepath: str,
    factor: float = 1.5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], float, float]:
    """
    Carrega CSV, remove outliers, separa X/y, normaliza, e faz train/test split.
    Retorna: X_train, X_test, y_train, y_test, feature_names, y_min, y_max
    """
    df = pd.read_csv(filepath)
    df = remove_outliers_iqr(df, factor)

    features = [c for c in df.columns if c not in ("ID", "Price")]
    X_raw = df[features].values
    y_raw = df["Price"].values.reshape(-1, 1)

    # Normalização Min-Max
    X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
    X = (X_raw - X_min) / (X_max - X_min)
    y_min, y_max = y_raw.min(), y_raw.max()
    y = (y_raw - y_min) / (y_max - y_min)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, features, y_min, y_max