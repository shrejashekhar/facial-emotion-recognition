import numpy as np

def add_batch_dim(tensor: np.ndarray) -> np.ndarray:
    return np.expand_dims(tensor, axis=0)
