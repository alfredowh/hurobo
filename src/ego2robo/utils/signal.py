"""Smoothing and resampling utilities for noisy per-frame trajectories."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def smooth_trajectory(positions: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    """Apply a Savitzky-Golay filter along the time axis of a trajectory.

    Chosen over a moving average because it preserves the shape of the
    approach/lift arcs (PLAN.md §4 step 3) rather than lagging and rounding them.

    Args:
        positions: shape (N, D), one row per timestep.
        window_length: odd number of samples in the filter window. Reduced
            automatically (to the largest valid odd number) if N is too short.
        polyorder: polynomial order fit within each window; must be < window_length.

    Returns:
        shape (N, D), smoothed trajectory.
    """
    n = positions.shape[0]
    effective_window = min(window_length, n if n % 2 == 1 else n - 1)
    if effective_window <= polyorder:
        return positions.copy()
    return savgol_filter(positions, window_length=effective_window, polyorder=polyorder, axis=0)


def fill_short_gaps(
    timestamps_s: np.ndarray, values: np.ndarray, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN runs no longer than `max_gap_s`; leave longer ones as NaN.

    Short dropouts (a missed MediaPipe detection, a motion-blurred frame) are
    safe to bridge. Long ones are not: interpolating across them would invent
    motion, so they stay NaN and the caller must decide to reject the clip
    (PLAN.md §7, "MediaPipe loses the hand during the grasp").

    Args:
        timestamps_s: shape (N,), strictly increasing timestamps in seconds.
        values: shape (N,) or (N, D); a row is missing if any entry is NaN.
        max_gap_s: longest missing run (measured between the valid samples
            bracketing it) that is still interpolated.

    Returns:
        (filled_values, still_missing): same shape as `values`, and a (N,) bool
        mask of rows that remain NaN. Missing runs touching either end of the
        clip are never filled (there is nothing to interpolate towards).
    """
    values_2d = values.reshape(len(values), -1).astype(np.float64)
    missing = np.isnan(values_2d).any(axis=1)
    filled = values_2d.copy()
    valid_idx = np.flatnonzero(~missing)
    if len(valid_idx) >= 2:
        for left, right in zip(valid_idx[:-1], valid_idx[1:], strict=True):
            if right - left > 1 and timestamps_s[right] - timestamps_s[left] <= max_gap_s:
                span = slice(left + 1, right)
                for d in range(filled.shape[1]):
                    filled[span, d] = np.interp(
                        timestamps_s[span], timestamps_s[[left, right]], filled[[left, right], d]
                    )
    still_missing = np.isnan(filled).any(axis=1)
    return filled.reshape(values.shape), still_missing


def resample_to_rate(
    timestamps_s: np.ndarray, values: np.ndarray, target_hz: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a (possibly irregular) time series to a fixed rate via linear interpolation.

    Needed because the phone's frame rate drifts slightly and MediaPipe drops
    occasional frames, while the sim's `WaypointFollower` (control.py) steps at
    a fixed 20 Hz (PLAN.md §4 step 7).

    Args:
        timestamps_s: shape (N,), strictly increasing timestamps in seconds.
        values: shape (N, D) or (N,), values at each timestamp.
        target_hz: output sample rate in Hz.

    Returns:
        (new_timestamps_s, new_values): new_timestamps_s has shape (M,) spaced
        at 1/target_hz starting at timestamps_s[0]; new_values has shape
        (M, D) or (M,) matching the input's dimensionality.
    """
    duration_s = timestamps_s[-1] - timestamps_s[0]
    num_samples = max(2, int(round(duration_s * target_hz)) + 1)
    new_timestamps_s = timestamps_s[0] + np.arange(num_samples) / target_hz
    new_timestamps_s = np.clip(new_timestamps_s, timestamps_s[0], timestamps_s[-1])
    interpolator = interp1d(timestamps_s, values, axis=0, kind="linear", assume_sorted=True)
    return new_timestamps_s, interpolator(new_timestamps_s)


def time_scale(timestamps_s: np.ndarray, factor: float) -> np.ndarray:
    """Stretch a time series by a constant factor (e.g. slow human motion for robot dynamics).

    Args:
        timestamps_s: shape (N,), timestamps in seconds, starting anywhere.
        factor: > 1 slows the motion down, < 1 speeds it up. PLAN.md §4 step 7
            uses roughly 1.5-2x to give the robot time to track waypoints.

    Returns:
        shape (N,), rescaled timestamps starting at the same offset as the input.
    """
    return timestamps_s[0] + (timestamps_s - timestamps_s[0]) * factor


def interpolate_pose_linear(fraction: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Linearly interpolate 3D positions between two endpoints (used for the carry segment).

    Args:
        fraction: shape (M,), values in [0, 1], 0 = pos_a, 1 = pos_b.
        pos_a: shape (3,), start position.
        pos_b: shape (3,), end position.

    Returns:
        shape (M, 3), interpolated positions.
    """
    fraction = fraction.reshape(-1, 1)
    return pos_a[None, :] * (1 - fraction) + pos_b[None, :] * fraction
