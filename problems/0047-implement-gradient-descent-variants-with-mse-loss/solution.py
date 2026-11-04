import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape

    w = np.array(weights, dtype=float).copy()

    if method == 'batch':
        curr_batch_size = m
    elif method == 'stochastic':
        curr_batch_size = 1
    elif method == 'mini_batch':
        curr_batch_size = batch_size
    else:
        raise ValueError(f"Unknown method: {method}")

    for _ in range(n_epochs):
        for i in range(0, m, curr_batch_size):
            X_batch = X[i : i + curr_batch_size]
            y_batch = y[i : i + curr_batch_size]

            y_pred = X_batch @ w
            error = y_pred - y_batch

            b = len(y_batch)
            gradient = (2.0 / b) * (X_batch.T @ error)

            w -= learning_rate * gradient

    return w