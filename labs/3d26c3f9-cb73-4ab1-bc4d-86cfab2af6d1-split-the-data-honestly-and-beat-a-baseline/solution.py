import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N = len(y)

    indices = np.random.RandomState(seed).permutation(N)

    N_train = int(N * train_frac)
    N_val = int(N * val_frac)

    train_idx = indices[:N_train]
    val_idx = indices[N_train : N_train + N_val]
    test_idx = indices[N_train + N_val :]

    mean = np.mean(y[train_idx])

    test_pred = np.full(shape=(len(test_idx), ), fill_value = mean)

    return test_pred, train_idx, val_idx, test_idx


