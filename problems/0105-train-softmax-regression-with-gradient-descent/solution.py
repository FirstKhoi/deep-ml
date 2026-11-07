
import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[list[float]], list[float]]:
    """
    Gradient-descent training algorithm for Softmax regression,
    optimizing parameters with Cross Entropy loss.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int).reshape(-1)

    m, n = X.shape
    C = np.max(y) + 1

    # 1. Add bias column
    X = np.column_stack((np.ones(m), X))

    # 2. Initialize weights
    w = np.zeros((C, n + 1))

    losses = []

    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    y_onehot = np.eye(C)[y]

    for _ in range(iterations):
        # 3. Forward
        z = X @ w.T
        probabilities = softmax(z)

        # 4. Calculate loss BEFORE update
        eps = 1e-15
        probabilities_clipped = np.clip(probabilities, eps, 1.0)

        loss = -np.sum(
            np.log(probabilities_clipped[np.arange(m), y])
        )
        losses.append(round(float(loss), 4))

        # 5. Gradient
        error = probabilities - y_onehot
        gradient = error.T @ X

        # 6. Update weights
        w -= learning_rate * gradient

    # 7. Return coefficients and losses
    coefficients = [
        [round(float(val), 4) for val in row]
        for row in w
    ]

    return coefficients, losses
