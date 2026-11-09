import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:

    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)

    splits = []

    for i in range(k):
        test_indices = folds[i]

        train_indices = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )

        splits.append((
            train_indices.tolist(),
            test_indices.tolist()
        ))

    return splits