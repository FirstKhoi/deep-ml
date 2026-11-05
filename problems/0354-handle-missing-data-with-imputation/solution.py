import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Your code here
    data = np.array(data, dtype=float).copy()

    for col in range(data.shape[1]):
        col_data = data[:, col]
        nan_mask = np.isnan(col_data)
        non_nan = col_data[~nan_mask]

        if len(non_nan) == 0:
            continue
    
        if strategy == "mean":
            fill_val = np.mean(non_nan)
        elif strategy == "median":
            fill_val = np.median(non_nan)
        elif strategy == "mode":
            vals, counts = np.unique(non_nan, return_counts=True)
            fill_val = vals[np.argmax(counts)]
        else:
            return -1

        data[nan_mask, col] = fill_val

    return data
        
