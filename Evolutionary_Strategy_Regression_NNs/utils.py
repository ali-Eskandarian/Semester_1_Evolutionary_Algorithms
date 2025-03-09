import numpy as np
from typing import List, Tuple
import pandas as pd

def load_and_preprocess_data(filename: str, normalize_: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the heart disease dataset
    """
    data = pd.read_csv(filename)
    X = data.drop(['num'], axis=1).values.astype(np.float64)
    y = data['num'].values.astype(np.float64)
    
    # Normalize features
    if normalize_:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X, y

