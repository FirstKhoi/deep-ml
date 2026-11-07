import numpy as np
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	logits = np.array(logits, dtype=float)
	
	numerator_exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))

	softmax = numerator_exp / np.sum(numerator_exp, axis=-1, keepdims=True)

	grad = softmax.copy()
	grad[target] -= 1.0

	return grad