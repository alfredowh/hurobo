"""Small statistics helpers for reporting success rates."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a success rate (95 % by default).

    Preferred over the normal approximation because benchmark cells have few episodes and rates near 0 or 1.

    Args:
        successes: number of successful episodes.
        trials: number of episodes.
        z: standard-normal quantile of the confidence level.

    Returns:
        (low, high) in [0, 1]; (0, 1) when there are no trials.
    """
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Start and stop (exclusive) index of the longest run of True in a boolean array.

    Args:
        mask: shape (N,) bool.

    Returns:
        (start, stop); (0, 0) when there is no True.
    """
    best, start = (0, 0), None
    for i, value in enumerate([*np.asarray(mask, dtype=bool), False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            best = (start, i) if i - start > best[1] - best[0] else best
            start = None
    return best
