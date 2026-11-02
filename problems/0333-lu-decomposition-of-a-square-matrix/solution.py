import numpy as np


def lu_decomposition(A: list) -> tuple:
  """Perform LU decomposition on a square matrix using Doolittle's method.

  Args:
      A: Square matrix as a list of lists

  Returns:
      tuple: (L, U) where L is lower triangular with 1s on diagonal,
             U is upper triangular, and A = L @ U
  """
  A = np.array(A, dtype=np.float64)
  n = A.shape[0]

  if A.shape != (n, n):
    raise ValueError("Matrix A must be square.")

  L = np.eye(n, dtype=np.float64)
  U = np.zeros((n, n), dtype=np.float64)

  for i in range(n):
    # 1. Compute Upper triangular matrix U row by row
    for j in range(i, n):
      s = sum(L[i, k] * U[k, j] for k in range(i))
      U[i, j] = A[i, j] - s

    # 2. Compute Lower triangular matrix L column by column
    for j in range(i + 1, n):
      s = sum(L[j, k] * U[k, i] for k in range(i))
      if np.isclose(U[i, i], 0.0):
        raise ValueError("Zero pivot encountered; LU decomposition fails.")
      L[j, i] = (A[j, i] - s) / U[i, i]

  return L, U