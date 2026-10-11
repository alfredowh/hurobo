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


def test_caption_scale_changes_text_size_but_not_the_default() -> None:
    import numpy as np

    from hurobo.utils.video import caption_frame

    blank = np.zeros((60, 400, 3), np.uint8)

    def lit_columns(image: np.ndarray) -> int:
        return int((image.max(axis=(0, 2)) > 0).sum())

    default = caption_frame(blank, "agentview (policy camera)")
    explicit = caption_frame(blank, "agentview (policy camera)", font_scale=0.7)
    smaller = caption_frame(blank, "agentview (policy camera)", font_scale=0.55)
    assert np.array_equal(default, explicit)  # other callers (push test, dashboards) are unaffected
    assert lit_columns(smaller) < lit_columns(default)
