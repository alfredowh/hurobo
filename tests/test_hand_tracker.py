"""HandTcpTracker on real hand frames (skipped when the gitignored sample frames are absent)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ego2robo.camera import CameraIntrinsics
from ego2robo.hand import HandTcpTracker
from ego2robo.utils.hand_landmarks import INDEX_TIP, THUMB_TIP

FRAMES_DIR = Path("data/raw/overview_gif/hand_moves_cube")


@pytest.mark.skipif(not FRAMES_DIR.exists(), reason="sample hand frames not present")
def test_pnp_depth_is_physical_on_every_detected_frame() -> None:
    """Regression: ITERATIVE PnP from the default guess gave z ~ 0 and 1000s of px error."""
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array([[800.0, 0, 400.0], [0, 800.0, 225.0], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        image_width=800,
        image_height=450,
    )
    results = []
    with HandTcpTracker(intrinsics) as tracker:
        for k, path in enumerate(sorted(FRAMES_DIR.glob("*.png"))):
            results.append(tracker.process_frame(cv2.imread(str(path)), timestamp_ms=100 * k))
    detected = [r for r in results if r is not None]
    assert len(detected) >= 0.8 * len(results)
    assert all(r.tcp_pos_cam[2] > 0.15 for r in detected)
    assert all(r.reprojection_error_px < 30.0 for r in detected)
    # The recorded landmarks are the PnP inputs: the aperture is the thumb-index distance between them.
    for r in detected:
        assert r.landmarks_hand.shape == (21, 3) and r.landmarks_image_px.shape == (21, 2)
        gap = np.linalg.norm(r.landmarks_hand[THUMB_TIP] - r.landmarks_hand[INDEX_TIP])
        assert gap == pytest.approx(r.pinch_aperture_m)
