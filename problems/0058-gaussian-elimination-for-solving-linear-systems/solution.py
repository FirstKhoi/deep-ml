import numpy as np


def gaussian_elimination(A, b):
    """Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix of shape (n, n)
    :param b: Right-hand side vector of shape (n,) or (n, 1)
    :return: Solution vector x of shape (n,)
    """
    # Convert to float to avoid integer truncation issues
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).reshape(-1).copy()

    n = len(b)
    if A.shape != (n, n):
        raise ValueError(
            f"Matrix A shape {A.shape} is incompatible with vector b of size {n}"
        )

    # 1. Forward Elimination with Partial Pivoting
    for i in range(n):
        # Find index of max pivot in column i (from row i down to n-1)
        pivot_idx = i + np.argmax(np.abs(A[i:, i]))

        if np.isclose(A[pivot_idx, i], 0.0):
            raise np.linalg.LinAlgError(
                "Matrix is singular or nearly singular to working precision."
            )

        # Swap rows in A and b if needed
        if pivot_idx != i:
            A[[i, pivot_idx]] = A[[pivot_idx, i]]
            b[[i, pivot_idx]] = b[[pivot_idx, i]]

        # Eliminate rows below current pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # 2. Back Substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]

    return x