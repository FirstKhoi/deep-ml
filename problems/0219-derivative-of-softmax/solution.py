import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	x = np.array(x, dtype=float)

	exp_scores = np.exp(x - np.max(x, axis=-1, keepdims=True))
	softmax = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
	
	res = np.diag(softmax) - np.outer(softmax, softmax)

	return res