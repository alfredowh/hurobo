"""VideoWriter produces browser-playable H.264 and rejects mismatched frames."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from hurobo.utils.video import VideoWriter, read_frames


def _codec(path: Path) -> str:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt"]
        + ["-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip()


def test_writes_h264_yuv420p_and_round_trips_frames(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, 20.0, (64, 48)) as writer:
        for value in (40, 120, 200):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    assert _codec(path) == "h264,yuv420p"
    frames = list(read_frames(path))
    assert len(frames) == 3
    assert frames[0].shape == (48, 64, 3)


def test_odd_frame_size_is_padded_not_rejected(tmp_path: Path) -> None:
    path = tmp_path / "odd.mp4"
    with VideoWriter(path, 10.0, (63, 47)) as writer:
        writer.write(np.zeros((47, 63, 3), dtype=np.uint8))
    assert _codec(path) == "h264,yuv420p"


def test_mismatched_frame_raises(tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "bad.mp4", 10.0, (64, 48)) as writer:
        with pytest.raises(ValueError):
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
