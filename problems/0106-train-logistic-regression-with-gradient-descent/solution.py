
import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1)

    m, d = X.shape

    # 1. Add bias column
    X = np.column_stack((np.ones(m), X))

    # 2. Initialize parameters
    theta = np.zeros(d + 1, dtype=float)

    losses = []

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    for _ in range(iterations):
        # 3. Forward
        z = X @ theta
        y_pred = sigmoid(z)

        # 4. BCE Loss (sum)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        loss = -np.sum(
            y * np.log(y_pred) +
            (1 - y) * np.log(1 - y_pred)
        )

        losses.append(round(float(loss), 4))

        # 5. Gradient
        error = y_pred - y
        gradient = X.T @ error

        # 6. Update
        theta -= learning_rate * gradient

    coefficients = [round(float(v), 4) for v in theta]

    return coefficients, losses
