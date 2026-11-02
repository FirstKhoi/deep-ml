import numpy as np


def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
  """Solve the system Ax = b using the Conjugate Gradient method.

  :param A: Symmetric positive-definite matrix
  :param b: Right-hand side vector
  :param n: Maximum number of iterations
  :param x0: Initial guess for solution (default is zero vector)
  :param tol: Convergence tolerance
  :return: Solution vector x
  """
  A = np.array(A, dtype=np.float64)
  b = np.array(b, dtype=np.float64).reshape(-1)

  # Initialize solution vector x
  if x0 is None:
    x = np.zeros_like(b, dtype=np.float64)
  else:
    x = np.array(x0, dtype=np.float64).reshape(-1)

  # Calculate initial residual vector r = b - A*x0
  r = b - A @ x
  p = r.copy()
  rs_old = np.dot(r, r)

  # Check if initial guess is already within tolerance
  if np.sqrt(rs_old) < tol:
    return x

  for _ in range(n):
    Ap = A @ p
    alpha = rs_old / np.dot(p, Ap)

    x += alpha * p
    r -= alpha * Ap

    rs_new = np.dot(r, r)

    # Check for convergence
    if np.sqrt(rs_new) < tol:
      break

    # Compute next search direction
    beta = rs_new / rs_old
    p = r + beta * p
    rs_old = rs_new

  return x