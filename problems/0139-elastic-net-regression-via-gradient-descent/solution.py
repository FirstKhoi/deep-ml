import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1)

    m, n = X.shape

    w = np.zeros(n)
    bias = 0.0

    for _ in range(max_iter):
        w_old = w.copy()
        bias_old = bias

        y_pred = X @ w + bias
        error = y_pred - y

        dL_dw = ((1 / m) * (X.T @ error) + alpha1 * np.sign(w) + 2 * alpha2 * w)

        dL_db = (1 / m) * np.sum(error)

        w -= learning_rate * dL_dw
        bias -= learning_rate * dL_db

        if (
            np.linalg.norm(dL_dw, ord=1) < tol
            and abs(bias - bias_old) < tol
        ):
            break

    return w, bias