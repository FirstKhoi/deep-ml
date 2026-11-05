import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    eps = 1e-8
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train,axis=0)
    
    std = np.where(std == 0, 1.0, std)
    X_test_scaled = (X_test - mean) / (std + eps)

    return X_test_scaled



    
