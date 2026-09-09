from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def load_coat_feature_matrix(
    path: str | Path,
    *,
    expected_rows: int | None = None,
) -> FloatArray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Coat feature matrix must be two-dimensional")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"expected {expected_rows} feature rows, got {matrix.shape[0]}"
        )
    if matrix.shape[1] == 0:
        raise ValueError("Coat feature matrix must contain at least one column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Coat feature matrix must be finite")
    return matrix


def load_coat_propensity_matrix(
    path: str | Path,
    *,
    n_users: int = 290,
    n_items: int = 300,
) -> FloatArray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (n_users, n_items):
        raise ValueError(
            "Coat propensity matrix must have shape "
            f"({n_users}, {n_items}), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Coat propensities must be finite")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("Coat propensities must lie in [0, 1]")
    return matrix
