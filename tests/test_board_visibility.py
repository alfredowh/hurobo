"""`board_visibility` finds a board in a clip and reports zero for a clip without one."""

from pathlib import Path

import cv2
import numpy as np

from hurobo.camera import ChArUcoCalibrator
from hurobo.scan import board_visibility


def _write_clip(path: Path, frame: np.ndarray, frames: int = 6) -> None:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def test_board_is_seen_in_a_clip_that_shows_it(tmp_path: Path) -> None:
    calibrator = ChArUcoCalibrator()
    board_image = calibrator.generate_board_image(pixels_per_square=60)
    frame = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    _write_clip(tmp_path / "board.mp4", frame)

    seen = board_visibility(tmp_path / "board.mp4", calibrator.board, step=2)

    assert seen.frames_sampled == 3
    assert seen.frames_with_markers == 3
    assert seen.max_markers == 17  # every marker of a 7 x 5 board


def test_blank_clip_reports_no_markers(tmp_path: Path) -> None:
    _write_clip(tmp_path / "blank.mp4", np.full((240, 320, 3), 127, dtype=np.uint8))

    seen = board_visibility(tmp_path / "blank.mp4", ChArUcoCalibrator().board)

    assert (seen.frames_with_markers, seen.max_markers) == (0, 0)
