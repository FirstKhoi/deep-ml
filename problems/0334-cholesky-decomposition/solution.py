import numpy as np


def cholesky_decomposition(A):
  """Perform Cholesky decomposition on a symmetric positive-definite matrix.

  Args:
      A: A symmetric positive-definite matrix (2D list or numpy array)

  Returns:
      L: Lower triangular matrix such that A = L @ L.T as a 2D list,
         or -1 if decomposition is not possible
  """
  A = np.array(A, dtype=np.float64)
  n = A.shape[0]

  if A.shape != (n, n):
    return -1

  L = np.zeros((n, n), dtype=np.float64)

  for i in range(n):
    for j in range(i + 1):
      s = sum(L[i, k] * L[j, k] for k in range(j))

      if i == j:
        # Diagonal elements
        val = A[i, i] - s
        if val <= 0:
          return -1
        L[i, j] = np.sqrt(val)
      else:
        # Off-diagonal elements
        L[i, j] = (A[i, j] - s) / L[j, j]

  return L.tolist()