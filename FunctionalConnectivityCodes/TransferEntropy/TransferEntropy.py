import numpy as np
from npeet import entropy_estimators as ee

def generate_lagged_vectors(signal: np.ndarray, lag: int) -> np.ndarray :
    """Generate lagged vectors from a signal."""
    n = len(signal)
    lagged_vectors = np.array([signal[i: n - lag + i + 1] for i in range(lag)]).T
    return lagged_vectors


def TransferEntropy(X: np.ndarray, Y: np.ndarray, d_x: int, d_y: int, w_x: int, w_y: int) -> float:
    """
    Calculate the Transfer Entropy from X to Y.
    
    Parameters:
    - Y (np.ndarray): The target signal.
    - X (np.ndarray): The source signal.
    - d_y (int): The lag for the target signal.
    - d_x (int): The lag for the source signal.
    - w_y (int): The window size for the target signal.
    - w_x (int): The window size for the source signal.

    Returns:
    - float: The calculated Transfer Entropy value.
    """

    X_lagged = generate_lagged_vectors(X, w_x)
    Y_lagged = generate_lagged_vectors(Y, w_y)
    X_lagged = X_lagged[:-d_x-1]
    Y_lagged = Y_lagged[:-d_y-1]

    # Remove the initial part where lagged vectors are not defined
    max_index = max(w_x + d_x , w_y + d_y)
    if w_x + d_x == max_index:
        Y_lagged = Y_lagged[max_index- w_y - d_y:]
    else:         
        X_lagged = X_lagged[max_index- w_x - d_x:]
    

    Y_t = Y[max_index:]
    
    return ee.cmi(Y_t, X_lagged, Y_lagged)
