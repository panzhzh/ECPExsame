from __future__ import annotations

import numpy as np


def confusion_matrix(gold: list[int], pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for g, p in zip(gold, pred):
        if 0 <= int(g) < num_classes and 0 <= int(p) < num_classes:
            matrix[int(g), int(p)] += 1
    return matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)

