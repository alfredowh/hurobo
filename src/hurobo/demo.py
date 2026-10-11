"""Human demonstration container and grasp/release segmentation from pinch aperture."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class HumanDemo:
    """One recorded, perception-processed human pick-place demonstration.

    All arrays share the same time axis (one row per sample, resampled to a
    fixed rate by `utils/signal.py`) and are expressed in the table frame
    (`T_table_*`), not the camera frame — the table marker is the world frame
    for the whole pipeline (see PLAN.md §4).

    Attributes:
        timestamps_s: shape (N,), seconds from the start of the episode.
        tcp_pos_table: shape (N, 3), hand TCP (thumb/index-tip midpoint) position.
        pinch_aperture_m: shape (N,), distance between thumb tip and index tip.
        is_grasping: shape (N,), bool, aperture thresholded with hysteresis.
        t_table_bowl: shape (4, 4), the bowl pose at the start of the episode
            (assumed static during one demo; re-detected per demo, not per frame).
        t_table_plate: shape (4, 4), the plate pose at the start of the episode.
        t_table_cam: shape (4, 4), the camera pose for this episode. Needed by
            `SceneTwin` (`twin.py`) to reproduce the real viewpoint in sim.
        source_video: path to the raw clip this demo was extracted from, kept
            for traceability from a generated sim demo back to its human source.
        grasp_frame_idx: index into the time axis of the detected grasp event,
            used by the anchor-correction step (PLAN.md §4 step 4). None until
            `GraspSegmenter` has run.
        release_frame_idx: index of the detected release event. None until segmented.
        source_start_time_s: time in `source_video` of sample 0 (the first frame with
            the hand in view), so exporters can fetch the matching frames.
    """

    timestamps_s: np.ndarray
    tcp_pos_table: np.ndarray
    pinch_aperture_m: np.ndarray
    is_grasping: np.ndarray
    t_table_bowl: np.ndarray
    t_table_plate: np.ndarray
    t_table_cam: np.ndarray
    source_video: Path
    grasp_frame_idx: int | None = field(default=None)
    release_frame_idx: int | None = field(default=None)
    source_start_time_s: float = 0.0

    def save(self, path: Path) -> None:
        """Save this demo to a single .npz file."""
        np.savez(
            path,
            timestamps_s=self.timestamps_s,
            tcp_pos_table=self.tcp_pos_table,
            pinch_aperture_m=self.pinch_aperture_m,
            is_grasping=self.is_grasping,
            t_table_bowl=self.t_table_bowl,
            t_table_plate=self.t_table_plate,
            t_table_cam=self.t_table_cam,
            source_video=str(self.source_video),
            grasp_frame_idx=-1 if self.grasp_frame_idx is None else self.grasp_frame_idx,
            release_frame_idx=-1 if self.release_frame_idx is None else self.release_frame_idx,
            source_start_time_s=self.source_start_time_s,
        )

    @classmethod
    def load(cls, path: Path) -> HumanDemo:
        """Load a demo previously saved with `save`."""
        data = np.load(path, allow_pickle=False)
        grasp_idx = int(data["grasp_frame_idx"])
        release_idx = int(data["release_frame_idx"])
        return cls(
            timestamps_s=data["timestamps_s"],
            tcp_pos_table=data["tcp_pos_table"],
            pinch_aperture_m=data["pinch_aperture_m"],
            is_grasping=data["is_grasping"],
            t_table_bowl=data["t_table_bowl"],
            t_table_plate=data["t_table_plate"],
            t_table_cam=data["t_table_cam"],
            source_video=Path(str(data["source_video"])),
            grasp_frame_idx=None if grasp_idx < 0 else grasp_idx,
            release_frame_idx=None if release_idx < 0 else release_idx,
            # Absent in files written before the field existed.
            source_start_time_s=float(data["source_start_time_s"]) if "source_start_time_s" in data else 0.0,
        )


class GraspSegmenter:
    """Detects grasp/release events from pinch aperture, with hysteresis.

    A single fixed threshold flickers near the boundary (e.g. fingers settling
    into a grasp). Two thresholds avoid that: the aperture must close past
    `close_threshold_m` to register a grasp, and open past the larger
    `open_threshold_m` to register a release.
    """

    def __init__(self, close_threshold_m: float = 0.03, open_threshold_m: float = 0.05) -> None:
        """Set the hysteresis thresholds.

        Args:
            close_threshold_m: aperture below this counts as "closed" (grasping).
            open_threshold_m: aperture above this counts as "open" (released).
                Must be >= close_threshold_m; the gap between them is the
                hysteresis band where the previous state is kept.
        """
        if open_threshold_m < close_threshold_m:
            raise ValueError("open_threshold_m must be >= close_threshold_m")
        self._close_threshold_m = close_threshold_m
        self._open_threshold_m = open_threshold_m

    def segment(self, pinch_aperture_m: np.ndarray) -> np.ndarray:
        """Compute a boolean is-grasping signal from a pinch-aperture time series.

        Args:
            pinch_aperture_m: shape (N,), distance between thumb and index tip.

        Returns:
            shape (N,) bool array, True while grasping.
        """
        is_grasping = np.zeros(len(pinch_aperture_m), dtype=bool)
        state = False
        for i, aperture in enumerate(pinch_aperture_m):
            state = self.next_state(float(aperture), state)
            is_grasping[i] = state
        return is_grasping

    def next_state(self, aperture_m: float, is_grasping: bool) -> bool:
        """One step of the hysteresis, for streaming use.

        Args:
            aperture_m: the newest thumb-index distance.
            is_grasping: the state after the previous sample.

        Returns:
            The state after this sample.
        """
        if not is_grasping and aperture_m < self._close_threshold_m:
            return True
        if is_grasping and aperture_m > self._open_threshold_m:
            return False
        return is_grasping

    @staticmethod
    def find_transition_frames(is_grasping: np.ndarray) -> tuple[int | None, int | None]:
        """Find the first grasp-onset and first release-onset frame indices.

        Args:
            is_grasping: shape (N,) bool, from `segment`.

        Returns:
            (grasp_frame_idx, release_frame_idx), each None if no transition
            of that kind occurs in the signal.
        """
        grasp_idx = None
        release_idx = None
        for i in range(1, len(is_grasping)):
            if is_grasping[i] and not is_grasping[i - 1] and grasp_idx is None:
                grasp_idx = i
            if not is_grasping[i] and is_grasping[i - 1] and grasp_idx is not None and release_idx is None:
                release_idx = i
        return grasp_idx, release_idx
