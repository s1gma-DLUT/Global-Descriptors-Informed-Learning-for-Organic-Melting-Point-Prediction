from typing import List, Tuple

import numpy as np


DEFAULT_SEED = 516


def build_random_folds(n_samples: int, n_folds: int = 5, seed: int = DEFAULT_SEED) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[:n_samples % n_folds] += 1

    folds: List[List[int]] = []
    start = 0
    for fold_size in fold_sizes:
        folds.append(indices[start:start + fold_size].tolist())
        start += fold_size
    return folds


def get_random_fold_indices(fold_indices: List[List[int]], fold: int, total_samples: int) -> Tuple[List[int], List[int]]:
    val_idx = sorted(fold_indices[fold - 1])
    val_idx_set = set(val_idx)
    train_idx = sorted(set(range(total_samples)) - val_idx_set)
    return train_idx, val_idx
