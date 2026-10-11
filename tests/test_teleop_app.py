"""Display layout of the teleoperation app."""

from __future__ import annotations

from hurobo.sim import LiberoTask
from hurobo.teleop_app import HUMAN, default_layout


def test_default_layout_keeps_stock_gallery_view_and_wrist_camera() -> None:
    twin = LiberoTask.TWIN_CAMERA_NAME
    assert twin not in ("galleryview", "agentview", "robot0_eye_in_hand")  # the stock views stay stock
    with_human = default_layout(True, twin)
    assert with_human[0] == [HUMAN, twin]
    assert with_human[1] == ["agentview", "galleryview", "robot0_eye_in_hand"]
    assert default_layout(False, twin) == [["agentview", "galleryview", "robot0_eye_in_hand"]]
