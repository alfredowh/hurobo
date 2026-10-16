"""Cleaning step before extraction: per-frame hand signals, frame and clip filters, and the saved cuts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.demo import GraspSegmenter
from hurobo.extract import MarkerEstimator
from hurobo.hand import HandTcpTracker
from hurobo.utils.video import get_video_fps, read_frames_with_timestamps

CUTS_FILE = "cuts.json"
ANALYSIS_DIR = ".analysis"
HAND_LANDMARKS = 21


@dataclass
class ClipSignals:
    """What the hand tracker and the marker detector saw in every frame of one clip.

    Attributes:
        video: the clip.
        times_s: shape (N,), frame times (the same clock the extraction uses).
        fps: nominal frame rate, for stepping in the UI.
        image_size: (width, height) in pixels.
        hand_count: shape (N,), hands the detector reported (0, 1, 2).
        aperture_m: shape (N,), thumb-index distance of the tracked hand, NaN where it could not be fitted.
        reprojection_px: shape (N,), PnP reprojection error of the fit, NaN where not fitted.
        tcp_depth_m: shape (N,), distance of the pinch point along the camera axis, NaN where not fitted.
        edge_margin_px: shape (N,), distance of the landmark closest to the image border (negative: outside),
            NaN where not fitted.
        landmarks_px: shape (N, 21, 2), image landmarks, NaN where not fitted.
        layout_seen: marker roles seen in the first second ("board", "bowl", "plate").
    """

    video: Path
    times_s: np.ndarray
    fps: float
    image_size: tuple[int, int]
    hand_count: np.ndarray
    aperture_m: np.ndarray
    reprojection_px: np.ndarray
    tcp_depth_m: np.ndarray
    edge_margin_px: np.ndarray
    landmarks_px: np.ndarray
    layout_seen: dict[str, bool]

    @property
    def fitted(self) -> np.ndarray:
        """shape (N,) bool, the hand was detected and its pose fitted."""
        return ~np.isnan(self.aperture_m)

    def save(self, path: Path) -> None:
        """Write the signals to an .npz, stamped with the clip's size and mtime (a changed clip is redone)."""
        stat = self.video.stat()
        np.savez_compressed(
            path,
            times_s=self.times_s,
            fps=self.fps,
            image_size=np.asarray(self.image_size),
            hand_count=self.hand_count,
            aperture_m=self.aperture_m,
            reprojection_px=self.reprojection_px,
            tcp_depth_m=self.tcp_depth_m,
            edge_margin_px=self.edge_margin_px,
            landmarks_px=self.landmarks_px.astype(np.float32),
            layout_seen=json.dumps(self.layout_seen),
            video_stamp=np.asarray([stat.st_size, stat.st_mtime_ns]),
        )

    @classmethod
    def load(cls, path: Path, video: Path) -> ClipSignals | None:
        """Read signals saved by `save`; None if missing or the clip changed since."""
        if not path.exists() or not video.exists():
            return None
        stat = video.stat()
        with np.load(path) as data:
            if tuple(data["video_stamp"]) != (stat.st_size, stat.st_mtime_ns):
                return None
            return cls(
                video=video,
                times_s=data["times_s"],
                fps=float(data["fps"]),
                image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
                hand_count=data["hand_count"],
                aperture_m=data["aperture_m"],
                reprojection_px=data["reprojection_px"],
                tcp_depth_m=data["tcp_depth_m"],
                edge_margin_px=data["edge_margin_px"],
                landmarks_px=data["landmarks_px"].astype(np.float64),
                layout_seen=json.loads(str(data["layout_seen"])),
            )


class ClipAnalyzer:
    """Runs the hand tracker (looking for two hands) and the layout markers over a clip."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        marker_estimator: MarkerEstimator | None = None,
        layout_window_s: float = 1.0,
    ) -> None:
        """Hold the camera and the optional marker estimator.

        Args:
            intrinsics: calibration of the phone at the clip's resolution.
            marker_estimator: used on the first `layout_window_s` only, to say whether the board and the
                bowl and plate markers were visible at the start; skipped if None.
            layout_window_s: how long the markers must be visible at the start.
        """
        self._intrinsics = intrinsics
        self._markers = marker_estimator
        self._layout_window_s = layout_window_s

    def analyze(self, video: Path) -> ClipSignals:
        """Per-frame signals of one clip.

        Args:
            video: the clip.
        """
        times, count, aperture, reprojection, depth, margin, landmarks = [], [], [], [], [], [], []
        seen = {"board": False, "bowl": False, "plate": False}
        width = height = 0
        nan_landmarks = np.full((HAND_LANDMARKS, 2), np.nan)
        with HandTcpTracker(self._intrinsics, num_hands=2) as tracker:
            for frame, t_s in read_frames_with_timestamps(video):
                height, width = frame.shape[:2]
                if self._markers is not None and t_s <= self._layout_window_s:
                    poses = self._markers.estimate(frame)
                    if poses is not None:
                        seen["board"] = True
                        seen["bowl"] |= "bowl" in poses
                        seen["plate"] |= "plate" in poses
                result = tracker.process_frame(frame, int(round(t_s * 1000)))
                times.append(t_s)
                count.append(tracker.last_hand_count)
                points = None if result is None else result.landmarks_image_px
                aperture.append(np.nan if result is None else result.pinch_aperture_m)
                reprojection.append(np.nan if result is None else result.reprojection_error_px)
                depth.append(np.nan if result is None else float(result.tcp_pos_cam[2]))
                landmarks.append(nan_landmarks if points is None else points)
                margin.append(
                    np.nan if points is None else float(np.min(np.minimum(points, [width, height] - points)))
                )
        return ClipSignals(
            video=video,
            times_s=np.asarray(times),
            fps=get_video_fps(video),
            image_size=(width, height),
            hand_count=np.asarray(count),
            aperture_m=np.asarray(aperture),
            reprojection_px=np.asarray(reprojection),
            tcp_depth_m=np.asarray(depth),
            edge_margin_px=np.asarray(margin),
            landmarks_px=np.stack(landmarks),
            layout_seen=seen if self._markers is not None else {},
        )

    def analyze_cached(self, video: Path, cache_dir: Path | None = None) -> ClipSignals:
        """`analyze`, reusing the result next to the clip while the clip is unchanged.

        Args:
            video: the clip.
            cache_dir: where the .npz goes; `<clip folder>/.analysis` by default.
        """
        cache_dir = cache_dir or video.parent / ANALYSIS_DIR
        path = cache_dir / f"{video.stem}.npz"
        cached = ClipSignals.load(path, video)
        if cached is not None:
            return cached
        signals = self.analyze(video)
        cache_dir.mkdir(parents=True, exist_ok=True)
        signals.save(path)
        return signals


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds of the frame filters and the clip filters.

    Attributes:
        max_hands: frames with more hands in view are dropped (the demo must be one hand).
        max_reprojection_px: frames whose hand fit is worse than this are dropped (the hand pose, and so the
            depth, is unreliable).
        min_edge_margin_px: frames where a landmark is closer than this to the image border are dropped (a
            partly visible hand is fitted badly).
        depth_range_m: frames whose pinch point lies outside this camera distance are dropped.
        max_gap_s: dropouts up to this long inside a kept stretch are bridged (extraction interpolates them).
        margin_after_release_s: how long after the release the clip is kept; later frames (a second grasp,
            the hand wandering off) are cut.
        min_closed_s: a closed pinch shorter than this is a flicker, not a grasp.
        min_open_s: an open pinch shorter than this between two closed ones is noise, not a release (the
            aperture estimate wobbles by about 1 cm, which is the width of the hysteresis band).
        min_duration_s: shorter kept clips are rejected.
        max_duration_s: longer kept clips are rejected.
        require_layout_markers: reject clips whose board, bowl or plate marker is missing in the first second.
    """

    max_hands: int = 1
    max_reprojection_px: float = 15.0
    min_edge_margin_px: float = 8.0
    depth_range_m: tuple[float, float] = (0.15, 3.0)
    max_gap_s: float = 0.4
    margin_after_release_s: float = 2.0
    min_closed_s: float = 0.2
    min_open_s: float = 0.3
    min_duration_s: float = 1.5
    max_duration_s: float = 30.0
    require_layout_markers: bool = True


class FrameFilter:
    """Says, per frame, why it cannot be used."""

    REASONS = ("no hand", "too many hands", "hand fit bad", "hand at image edge", "depth implausible")

    def __init__(self, config: CleaningConfig | None = None) -> None:
        """Hold the thresholds.

        Args:
            config: thresholds; defaults to `CleaningConfig()`.
        """
        self._config = config or CleaningConfig()

    def flags(self, signals: ClipSignals) -> dict[str, np.ndarray]:
        """One boolean array (N,) per reason in `REASONS`, True where that reason applies.

        Args:
            signals: the clip's per-frame signals.
        """
        cfg = self._config
        fitted = signals.fitted
        low, high = cfg.depth_range_m
        with np.errstate(invalid="ignore"):  # NaN where the hand was not fitted: every comparison is False
            return {
                "no hand": signals.hand_count == 0,
                "too many hands": signals.hand_count > cfg.max_hands,
                "hand fit bad": fitted & (signals.reprojection_px > cfg.max_reprojection_px),
                "hand at image edge": fitted & (signals.edge_margin_px < cfg.min_edge_margin_px),
                "depth implausible": fitted & ((signals.tcp_depth_m < low) | (signals.tcp_depth_m > high)),
            }

    def usable(self, signals: ClipSignals) -> np.ndarray:
        """shape (N,) bool, frames with a fitted single hand that passes every filter."""
        rejected = np.any(list(self.flags(signals).values()), axis=0)
        return signals.fitted & ~rejected


@dataclass
class ClipCut:
    """The decision for one clip.

    Attributes:
        start_s: first kept time (the clip's own clock), None if the clip is rejected without a window.
        end_s: last kept time.
        accepted: use the clip in extraction.
        source: "auto" (proposed by `ClipCleaner`) or "manual" (edited in the UI).
        reasons: why the clip was rejected (empty if accepted).
        notes: what was cut and why, for the UI.
    """

    start_s: float | None
    end_s: float | None
    accepted: bool
    source: str = "auto"
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def window_s(self) -> tuple[float, float] | None:
        """(start, end) for the extraction, None if the clip has no window."""
        if self.start_s is None or self.end_s is None:
            return None
        return self.start_s, self.end_s


class ClipCleaner:
    """Proposes the stretch of a clip to keep and whether the clip is usable at all."""

    def __init__(self, config: CleaningConfig | None = None, segmenter: GraspSegmenter | None = None) -> None:
        """Hold thresholds.

        Args:
            config: thresholds; defaults to `CleaningConfig()`.
            segmenter: pinch thresholds; defaults to 3 cm close, 5 cm open (what the extraction uses).
        """
        self._config = config or CleaningConfig()
        self._segmenter = segmenter or GraspSegmenter()
        self._frames = FrameFilter(self._config)

    @property
    def config(self) -> CleaningConfig:
        """The thresholds in use."""
        return self._config

    @property
    def segmenter(self) -> GraspSegmenter:
        """The pinch thresholds in use."""
        return self._segmenter

    def closed_signal(self, signals: ClipSignals) -> np.ndarray:
        """shape (N,) bool, the pinch is closed: hysteresis over the aperture, held through dropouts, with
        open flickers shorter than `min_open_s` filled in."""
        aperture = signals.aperture_m
        filled = np.where(np.isnan(aperture), np.nan, aperture)
        valid = ~np.isnan(filled)
        if not valid.any():
            return np.zeros(len(aperture), dtype=bool)
        filled = np.interp(np.arange(len(aperture)), np.flatnonzero(valid), filled[valid])
        closed = self._segmenter.segment(filled)
        return _bridge_gaps(closed, int(round(self._config.min_open_s * signals.fps)))

    def propose(self, signals: ClipSignals) -> ClipCut:
        """Cut a clip down to one clean stretch: a single hand, open, then one close and one open again.

        The stretch is the first gap-free run of usable frames that contains a complete grasp cycle. Its start
        is the first usable frame (or the end of a pinch that was already closed on entry); its end is
        `margin_after_release_s` after the release. Everything before, after and in between is cut.

        Args:
            signals: the clip's per-frame signals.

        Returns:
            The proposed cut; `accepted` is False, with `reasons`, if no stretch qualifies or a filter fails.
        """
        cfg = self._config
        times = signals.times_s
        notes: list[str] = []
        usable = self._frames.usable(signals)
        if not usable.any():
            return ClipCut(None, None, False, reasons=["no frame with a single, well-fitted hand"])
        bridged = _bridge_gaps(usable, int(round(cfg.max_gap_s * signals.fps)))
        closed = self.closed_signal(signals)
        cycle = self._first_cycle(bridged, closed, signals)
        if cycle is None:
            return ClipCut(
                None,
                None,
                False,
                reasons=["no stretch with one hand where the pinch opens, closes and opens again"],
                notes=[f"usable frames: {usable.mean():.0%}"],
            )
        first, last_release, stretch_end, extra_cycles = cycle
        start = first
        end = min(stretch_end, last_release + int(round(cfg.margin_after_release_s * signals.fps)))
        if start > 0:
            notes.append(f"cut {times[start] - times[0]:.2f} s at the start (no usable hand or pinch closed)")
        if end < len(times) - 1:
            notes.append(f"cut {times[-1] - times[end]:.2f} s at the end")
        if extra_cycles:
            notes.append(f"{extra_cycles} further grasp(s) after the first release dropped")
        reasons = self._clip_reasons(signals, start, end)
        return ClipCut(float(times[start]), float(times[end]), not reasons, reasons=reasons, notes=notes)

    def _first_cycle(
        self, bridged: np.ndarray, closed: np.ndarray, signals: ClipSignals
    ) -> tuple[int, int, int, int] | None:
        """(start, release frame, stretch end, later grasps) of the first usable stretch with a full cycle."""
        min_closed = int(round(self._config.min_closed_s * signals.fps))
        for seg_start, seg_stop in _runs(bridged):
            runs = [(a, b) for a, b in _runs(closed[seg_start:seg_stop]) if b - a >= min_closed]
            # A pinch closed on entry has no open frames before it: it is not the grasp, only trimmed.
            runs = [(seg_start + a, seg_start + b) for a, b in runs]
            entry_closed = [run for run in runs if run[0] == seg_start]
            candidates = [run for run in runs if run[0] > seg_start and run[1] < seg_stop]
            if not candidates:
                continue
            start = entry_closed[0][1] if entry_closed else seg_start
            grasp, release = candidates[0]
            return start, release, seg_stop - 1, len(candidates) - 1
        return None

    def _clip_reasons(self, signals: ClipSignals, start: int, end: int) -> list[str]:
        cfg = self._config
        reasons: list[str] = []
        duration = float(signals.times_s[end] - signals.times_s[start])
        if duration < cfg.min_duration_s:
            reasons.append(f"kept stretch is only {duration:.1f} s (min {cfg.min_duration_s} s)")
        if duration > cfg.max_duration_s:
            reasons.append(f"kept stretch is {duration:.1f} s (max {cfg.max_duration_s} s)")
        if cfg.require_layout_markers and signals.layout_seen:
            missing = [name for name, seen in signals.layout_seen.items() if not seen]
            if missing:
                reasons.append(f"{', '.join(missing)} marker not seen in the first second")
        return reasons


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of every run of True in a boolean array."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _bridge_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Set False runs of at most `max_gap_frames` that sit between True frames to True."""
    bridged = np.asarray(mask, dtype=bool).copy()
    for start, stop in _runs(~bridged):
        if start > 0 and stop < len(bridged) and stop - start <= max_gap_frames:
            bridged[start:stop] = True
    return bridged


class CutStore:
    """The cuts of a folder of clips, in `<folder>/cuts.json`."""

    def __init__(self, path: Path) -> None:
        """Load the file if it exists.

        Args:
            path: the cuts file.
        """
        self.path = path
        self._cuts: dict[str, ClipCut] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self._cuts = {name: ClipCut(**entry) for name, entry in raw.get("clips", {}).items()}

    @classmethod
    def for_folder(cls, folder: Path) -> CutStore:
        """The store next to a folder's clips."""
        return cls(folder / CUTS_FILE)

    def get(self, clip: Path) -> ClipCut | None:
        """The stored cut of a clip, None if there is none."""
        return self._cuts.get(clip.name)

    def set(self, clip: Path, cut: ClipCut) -> None:
        """Store a cut and write the file."""
        self._cuts[clip.name] = cut
        self.save()

    def remove(self, clip: Path) -> None:
        """Forget a clip's cut and write the file."""
        self._cuts.pop(clip.name, None)
        self.save()

    def save(self) -> None:
        """Write the file (clips sorted by name)."""
        payload = {"version": 1, "clips": {k: asdict(v) for k, v in sorted(self._cuts.items())}}
        self.path.write_text(json.dumps(payload, indent=2))

    def names(self) -> list[str]:
        """Clip file names with a stored cut."""
        return sorted(self._cuts)


VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")


def find_clips(location: str) -> list[Path]:
    """Clips at a folder, a glob pattern or a single file path (sorted by name).

    Args:
        location: e.g. "data/raw/e1", "data/raw/e1/clip_*.mp4" or "data/raw/e1/clip_01.mp4"; `~` is expanded.
    """
    import glob

    path = Path(location).expanduser()
    if path.is_dir():
        found = [p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
    else:
        found = [Path(p) for p in glob.glob(str(path)) if Path(p).suffix.lower() in VIDEO_SUFFIXES]
    return sorted(found)
