import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = np.array(scores, float)
    
    numerator = np.exp(scores - np.max(scores, axis=-1, keepdims=True))

    return numerator / np.sum(numerator, axis=-1, keepdims=True)