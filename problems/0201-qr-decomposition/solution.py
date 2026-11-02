import numpy as np


def qr_decomposition(
    A: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
  """Perform QR decomposition using Gram-Schmidt process.

  Args:
      A: An m x n matrix represented as list of lists

  Returns:
      Tuple of (Q, R) where Q is orthogonal and R is upper triangular
  """
  A_arr = np.array(A, dtype=np.float64)
  m, n = A_arr.shape

  Q = np.zeros((m, n), dtype=np.float64)
  R = np.zeros((n, n), dtype=np.float64)

  for j in range(n):
    v = A_arr[:, j].copy()
    for i in range(j):
      # Compute projection coefficient R[i, j] = Q_i^T * A_j
      R[i, j] = np.dot(Q[:, i], A_arr[:, j])
      v -= R[i, j] * Q[:, i]

    # Compute norm for diagonal entry R[j, j]
    norm = np.linalg.norm(v)
    R[j, j] = norm

    # Normalize vector to form the orthonormal column of Q
    if not np.isclose(norm, 0.0):
      Q[:, j] = v / norm
    else:
      Q[:, j] = 0.0  # Handle linearly dependent columns gracefully

  return Q.tolist(), R.tolist()