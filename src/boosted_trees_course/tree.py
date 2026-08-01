"""CART node, split, fit, and prediction code built during Lessons 2–5."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SplitResult:
    """Statistics describing the best split at one node."""

    threshold: float
    left_prediction: float
    right_prediction: float
    left_loss: float
    right_loss: float
    total_loss: float


def midpoint_thresholds(
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return midpoints between sorted distinct feature values."""
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional feature array")

    unique_x = np.unique(x)

    if unique_x.size < 2:
        return np.empty(0, dtype=np.float64)

    return (unique_x[:-1] + unique_x[1:]) / 2.0


def squared_error(y: NDArray[np.float64]) -> float:
    """Return RSS around the mean of a non-empty target array."""
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional target array")
    if y.size == 0:
        raise ValueError("y must contain at least one target")

    prediction = float(y.mean())
    residuals = y - prediction
    return float(np.sum(residuals**2))


def best_split_1d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> SplitResult:
    """Return the minimum-RSS split for one feature at one node."""
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional feature array")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional target array")
    if x.size != y.size:
        raise ValueError("x and y must contain the same number of observations")
    if x.size == 0:
        raise ValueError("x and y must not be empty")

    thresholds = midpoint_thresholds(x)

    if thresholds.size == 0:
        raise ValueError("x must contain at least two distinct values")

    best_result: SplitResult | None = None

    for threshold in thresholds:
        left_mask = x <= threshold
        right_mask = x > threshold

        left_targets = y[left_mask]
        right_targets = y[right_mask]

        left_prediction = float(left_targets.mean())
        right_prediction = float(right_targets.mean())

        left_loss = squared_error(left_targets)
        right_loss = squared_error(right_targets)
        total_loss = left_loss + right_loss

        candidate = SplitResult(
            threshold=float(threshold),
            left_prediction=left_prediction,
            right_prediction=right_prediction,
            left_loss=left_loss,
            right_loss=right_loss,
            total_loss=total_loss,
        )

        if best_result is None or candidate.total_loss < best_result.total_loss:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("no valid split candidate was evaluated")

    return best_result
