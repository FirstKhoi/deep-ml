import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    m = X.shape[0]
    W = W.copy().astype(float)
    b = float(b)
    
    epochs = 500
    lr = 0.05
    for _ in range(epochs):
        y_pred = X @ W + b
        error = y_pred - y
        
        # Gradient
        grad_W = (1.0 / m) * (X.T @ error)
        grad_b = (1.0 / m) * np.sum(error)
        
        # Update weights
        W -= lr * grad_W
        b -= lr * grad_b
        
    return W, float(b)