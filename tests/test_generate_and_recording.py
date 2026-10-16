"""Dataset frame conversion, the printable pad, the session command list and layout feasibility."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hurobo.generate import STATE_NAMES, frame_from_obs, quat_xyzw_to_axis_angle, sim_features
from hurobo.recording import RecordingSpec, render_bowl_pad, render_plate_marker, session_commands
from hurobo.scaled_native import Workspace, layout_feasible_fraction


def test_axis_angle_matches_the_lerobot_formula() -> None:
    # Rotation of 90 degrees about z: quaternion (0, 0, sin 45, cos 45).
    result = quat_xyzw_to_axis_angle(np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]))
    assert np.allclose(result, [0.0, 0.0, np.pi / 2])
    assert np.allclose(quat_xyzw_to_axis_angle(np.array([0.0, 0.0, 0.0, 1.0])), 0.0)
    # No wrapping to [0, pi]: w < 0 gives an angle above pi, as in the evaluation-time processor.
    flipped = quat_xyzw_to_axis_angle(np.array([1.0, 0.0, 0.0, -0.05]) / np.linalg.norm([1, 0, 0, -0.05]))
    assert np.linalg.norm(flipped) > np.pi


def test_frame_has_the_dataset_schema_and_upright_images() -> None:
    image = np.zeros((1, 256, 256, 3), np.uint8)
    image[0, 0, 0] = 255  # top-left pixel of the raw (upside-down) env image
    obs = {
        "pixels": {"image": image, "image2": image.copy()},
        "robot_state": {
            "eef": {"pos": np.array([[0.1, 0.2, 0.3]]), "quat": np.array([[0.0, 0.0, 0.0, 1.0]])},
            "gripper": {"qpos": np.array([[0.04, -0.04]])},
        },
    }
    frame = frame_from_obs(obs, np.arange(7))
    features = sim_features()
    assert set(frame) == set(features)
    assert frame["observation.state"].shape == (len(STATE_NAMES),)
    assert frame["action"].dtype == np.float32
    # Turned by 180 degrees: the raw top-left pixel ends up bottom-right.
    assert frame["observation.images.image"][-1, -1].tolist() == [255, 255, 255]


def test_bowl_pad_has_the_marker_and_a_circle_of_the_bowl_diameter() -> None:
    spec = RecordingSpec()
    dpi = 300
    pad = render_bowl_pad(spec, 0.138, dpi=dpi)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(pad)
    assert ids is not None and ids.ravel().tolist() == [spec.bowl_marker_id]
    side_px = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
    assert side_px / (dpi / 0.0254) == pytest.approx(spec.marker_length_m, abs=0.001)
    # The circle: on the centre row, past the marker, the outermost dark pixels are the bowl diameter apart.
    marker_right_px = int(corners[0][0][:, 0].max()) + 5
    dark = np.flatnonzero(pad[pad.shape[0] // 2, marker_right_px:] < 128)
    diameter_m = (dark[-1] - dark[0]) / (dpi / 0.0254)
    assert diameter_m == pytest.approx(0.138, abs=0.003)


def test_bowl_pad_that_does_not_fit_raises() -> None:
    with pytest.raises(ValueError):
        render_bowl_pad(RecordingSpec(), 0.30)


def test_plate_marker_sheet_is_detected() -> None:
    sheet = render_plate_marker(RecordingSpec())
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    _, ids, _ = detector.detectMarkers(sheet)
    assert ids is not None and ids.ravel().tolist() == [22]


def test_session_commands_are_ordered_and_use_the_given_paths() -> None:
    commands = session_commands(clips=12, generated=50)
    scripts = [command.split("scripts/")[1].split(".py")[0] for _, command in commands]
    assert scripts == [
        "16_plan_recording",
        "17_check_capture",
        "20_clean_clips",
        "02_extract_demos",
        "11_export_human_dataset",
        "03_replay",
        "04_generate_dataset",
    ]
    assert "--clips 12" in commands[0][1] and "--count 50" in commands[-1][1]


def test_distractors_shrink_the_feasible_layouts() -> None:
    workspace = Workspace()
    cleared = layout_feasible_fraction(workspace, 0.069, 0.1275, {}, samples=20000)
    blocked = layout_feasible_fraction(
        workspace, 0.069, 0.1275, {"cabinet": (-0.10, 0.10, -0.15, 0.15)}, samples=20000
    )
    assert 0.1 < cleared < 1.0
    assert blocked < cleared


def test_longest_true_run_finds_the_longest_dropout() -> None:
    from hurobo.utils.stats import longest_true_run

    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert longest_true_run(mask) == (4, 7)
    assert longest_true_run(np.zeros(3, dtype=bool)) == (0, 0)
    assert longest_true_run(np.ones(3, dtype=bool)) == (0, 3)
