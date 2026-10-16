"""Tests for the clip-cleaning step: frame filters, the proposed cut and the cuts file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hurobo.clip_clean import CleaningConfig, ClipCleaner, ClipCut, ClipSignals, CutStore, FrameFilter

FPS = 20.0
HALF_SECOND = CleaningConfig(margin_after_release_s=0.5)  # short, so the small test series are not all margin


def _signals(
    aperture_cm: list[float | None],
    hands: list[int] | None = None,
    layout: dict[str, bool] | None = None,
    reprojection: float = 3.0,
) -> ClipSignals:
    """Signals from an aperture series in cm (None = hand not fitted)."""
    n = len(aperture_cm)
    aperture = np.array([np.nan if a is None else a / 100 for a in aperture_cm])
    fitted = ~np.isnan(aperture)
    counts = np.array(hands) if hands is not None else fitted.astype(int)
    return ClipSignals(
        video=Path("clip.mp4"),
        times_s=np.arange(n) / FPS,
        fps=FPS,
        image_size=(1920, 1080),
        hand_count=counts,
        aperture_m=aperture,
        reprojection_px=np.where(fitted, reprojection, np.nan),
        tcp_depth_m=np.where(fitted, 0.8, np.nan),
        edge_margin_px=np.where(fitted, 200.0, np.nan),
        landmarks_px=np.full((n, 21, 2), np.nan),
        layout_seen=layout if layout is not None else {},
    )


def _cycle(open_n: int, closed_n: int, open_after_n: int) -> list[float | None]:
    return [8.0] * open_n + [1.0] * closed_n + [8.0] * open_after_n


def test_leading_and_trailing_frames_without_a_hand_are_cut() -> None:
    series = [None] * 10 + _cycle(10, 10, 20) + [None] * 15
    cut = ClipCleaner(HALF_SECOND).propose(_signals(series))
    assert cut.accepted
    assert cut.start_s == 10 / FPS
    assert cut.end_s == 40 / FPS  # 0.5 s after the release at frame 30


def test_only_the_first_close_and_open_is_kept() -> None:
    series = _cycle(10, 10, 15) + _cycle(0, 10, 15)
    cut = ClipCleaner(HALF_SECOND).propose(_signals(series))
    release = 20
    assert cut.accepted
    assert cut.end_s == (release + 10) / FPS  # 0.5 s after the release
    assert any("dropped" in note for note in cut.notes)


def test_a_pinch_closed_on_entry_is_trimmed_not_used_as_the_grasp() -> None:
    series = [1.0] * 8 + _cycle(10, 10, 20)
    cut = ClipCleaner(HALF_SECOND).propose(_signals(series))
    assert cut.accepted
    assert cut.start_s == 8 / FPS


def test_two_hands_split_the_clip_and_the_stretch_with_the_cycle_is_used() -> None:
    series = _cycle(5, 5, 5) + _cycle(10, 10, 20)
    hands = [2] * 15 + [1] * 40
    cut = ClipCleaner(HALF_SECOND).propose(_signals(series, hands=hands))
    assert cut.accepted
    assert cut.start_s == 15 / FPS


def test_a_clip_without_a_release_is_rejected() -> None:
    cut = ClipCleaner(HALF_SECOND).propose(_signals([8.0] * 20 + [1.0] * 30))
    assert not cut.accepted
    assert cut.window_s is None


def test_short_dropouts_are_bridged_but_long_ones_split_the_clip() -> None:
    bridged = _cycle(10, 10, 20)
    bridged[4:6] = [None, None]  # 0.1 s
    assert ClipCleaner(HALF_SECOND).propose(_signals(bridged)).start_s == 0.0
    split = _cycle(10, 10, 20)
    split[2:14] = [None] * 12  # 0.6 s, longer than max_gap_s: the grasp is no longer in the first stretch
    assert (
        ClipCleaner(HALF_SECOND).propose(_signals(split)).start_s != 0.0
        or not ClipCleaner(HALF_SECOND).propose(_signals(split)).accepted
    )


def test_missing_layout_markers_reject_the_clip() -> None:
    signals = _signals([None] * 4 + _cycle(10, 10, 20), layout={"board": True, "bowl": True, "plate": False})
    cut = ClipCleaner(HALF_SECOND).propose(signals)
    assert not cut.accepted
    assert any("plate" in reason for reason in cut.reasons)


def test_frame_filter_flags_bad_fits_and_edges() -> None:
    signals = _signals(_cycle(5, 5, 5), reprojection=40.0)
    flags = FrameFilter().flags(signals)
    assert flags["hand fit bad"].all()
    assert not FrameFilter().usable(signals).any()


def test_cuts_round_trip_through_the_json_file(tmp_path: Path) -> None:
    store = CutStore.for_folder(tmp_path)
    clip = tmp_path / "clip_01.mp4"
    store.set(clip, ClipCut(1.0, 5.5, True, source="manual", notes=["trimmed by hand"]))
    reloaded = CutStore.for_folder(tmp_path).get(clip)
    assert reloaded is not None and reloaded.window_s == (1.0, 5.5) and reloaded.source == "manual"
    assert CutStore.for_folder(tmp_path).get(tmp_path / "other.mp4") is None


def test_a_short_open_flicker_inside_a_grasp_is_not_a_release() -> None:
    series = _cycle(10, 10, 0) + [6.0, 2.0] + [1.0] * 10 + [8.0] * 20  # one noisy sample above 5 cm
    cut = ClipCleaner(HALF_SECOND).propose(_signals(series))
    assert cut.accepted
    assert cut.end_s == (10 + 10 + 2 + 10 + 10) / FPS  # 0.5 s after the real release at frame 32


def test_the_default_keeps_two_seconds_after_the_release_but_not_past_the_stretch() -> None:
    release = 20
    long_tail = _signals(_cycle(10, 10, 80))  # 4 s of open hand after the release
    assert ClipCleaner().propose(long_tail).end_s == (release + 40) / FPS
    short_tail = _signals(_cycle(10, 10, 15))  # the hand leaves 0.75 s after the release
    assert ClipCleaner().propose(short_tail).end_s == (10 + 10 + 15 - 1) / FPS
