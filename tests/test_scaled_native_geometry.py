"""Frame map, layout sampling, recording plan and retargeting geometry for the scaled-native experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hurobo.camera import CameraIntrinsics
from hurobo.capture_plan import HAND_STARTS, CameraPlacementSearch, CapturePlanner, camera_pose_looking_at
from hurobo.real_setup import RealSetup
from hurobo.retarget import ObjectCentricRetargeter, resample_path, wrap_symmetric_yaw
from hurobo.scaled_native import LayoutSampler, SimLayout, Workspace, table_to_world_map
from hurobo.twin import SIM_TABLE_TOP_Z
from hurobo.utils.synthetic import synthetic_pick_place_demo

MEASUREMENTS = """
table: {width_m: 1.445, depth_m: 0.60, left_edge_x_m: -0.03, right_edge_x_m: 1.415,
        top_edge_y_m: 0.02, near_edge_y_m: -0.58, centre_xy_m: [0.6925, -0.28]}
charuco: {squares_x: 7, squares_y: 5, square_length_m: 0.025, marker_length_m: 0.0186,
          tablet_thickness_m: 0.008}
plate: {diameter_m: 0.255, height_m: 0.028}
bowl: {diameter_m: 0.138, height_m: 0.067}
"""


@pytest.fixture
def setup(tmp_path: Path) -> RealSetup:
    path = tmp_path / "measurements.yaml"
    path.write_text(MEASUREMENTS)
    return RealSetup.load(path)


def test_setup_is_read_from_yaml(setup: RealSetup) -> None:
    assert setup.plate.radius_m == pytest.approx(0.1275)
    assert setup.bowl.height_m == pytest.approx(0.067)
    assert setup.board.width_m == pytest.approx(0.175)
    assert np.allclose(setup.table.centre_xy_m, [0.6925, -0.28])


def test_table_to_world_map_puts_reach_direction_along_robot_x(setup: RealSetup) -> None:
    workspace = Workspace()
    mapping = table_to_world_map(setup, workspace)
    centre = mapping.point_to_world(np.array([*setup.table.centre_xy_m, 0.0]))
    assert np.allclose(centre[:2], workspace.centre) and centre[2] == pytest.approx(SIM_TABLE_TOP_Z)
    away = mapping.point_to_world(np.array([*setup.table.centre_xy_m, 0.0]) + [0.0, 0.1, 0.0]) - centre
    right = mapping.point_to_world(np.array([*setup.table.centre_xy_m, 0.0]) + [0.1, 0.0, 0.0]) - centre
    assert np.allclose(away, [0.1, 0.0, 0.0], atol=1e-9)  # away from the person is away from the robot base
    assert np.allclose(right, [0.0, -0.1, 0.0], atol=1e-9)  # the person's right is world -y


def test_sampled_layouts_do_not_overlap_and_stay_inside(setup: RealSetup) -> None:
    workspace = Workspace()
    sampler = LayoutSampler(workspace, setup.bowl.radius_m, setup.plate.radius_m, seed=1)
    for _ in range(100):
        layout = sampler.sample()
        assert workspace.contains(layout.bowl_xy) and workspace.contains(layout.plate_xy)
        assert np.linalg.norm(layout.bowl_xy - layout.plate_xy) >= sampler.min_centre_distance_m


def test_workspace_too_small_for_the_objects_raises(setup: RealSetup) -> None:
    sampler = LayoutSampler(Workspace((0.0, 0.1), (0.0, 0.1)), 0.069, 0.1275)
    with pytest.raises(RuntimeError):
        sampler.sample(max_tries=50)


def test_planned_clips_are_valid_sim_layouts_and_balanced(setup: RealSetup) -> None:
    workspace = Workspace()
    mapping = table_to_world_map(setup, workspace)
    clips = CapturePlanner(setup, workspace, seed=0).plan(12)
    assert [c.index for c in clips] == list(range(1, 13))
    for clip in clips:
        bowl_w = mapping.point_to_world(np.array([*clip.bowl_xy_table, 0.0]))[:2]
        plate_w = mapping.point_to_world(np.array([*clip.plate_xy_table, 0.0]))[:2]
        assert workspace.contains(bowl_w) and workspace.contains(plate_w)
        assert np.linalg.norm(bowl_w - plate_w) >= setup.bowl.radius_m + setup.plate.radius_m
    assert sum(c.grasp_side == "left" for c in clips) == 6
    assert {c.hand_start.name for c in clips} == {h.name for h in HAND_STARTS}


def test_camera_search_finds_poses_that_see_board_and_workspace(setup: RealSetup) -> None:
    intrinsics = CameraIntrinsics.pinhole(1920, 1080, 1310.0)
    search = CameraPlacementSearch(setup, Workspace(), intrinsics, min_board_width_px=250.0)
    placements = search.search(step_m=0.2)
    assert placements, "no pose sees the board at 250 px and the whole work area"
    best = placements[0]
    assert best.board_width_px >= 250.0 and best.min_margin > 0.02 and best.elevation_deg > 20
    table = setup.table
    inside = (table.left_edge_x_m < best.position_table[0] < table.right_edge_x_m) and (
        table.near_edge_y_m < best.position_table[1] < table.far_edge_y_m
    )
    assert not inside, "the tripod must not stand on the table"


def test_camera_pose_looks_at_its_aim() -> None:
    pose = camera_pose_looking_at(np.array([0.0, -1.0, 0.8]), np.array([0.0, 0.0, 0.0]))
    forward = pose[:3, 2]
    expected = np.array([0.0, 1.0, -0.8]) / np.linalg.norm([0.0, 1.0, -0.8])
    assert np.allclose(forward, expected)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)


def test_resample_path_keeps_end_points_and_spacing() -> None:
    path = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]])
    out = resample_path(path, 0.025)
    assert np.allclose(out[0], path[0]) and np.allclose(out[-1], path[-1])
    assert np.linalg.norm(np.diff(out, axis=0), axis=1).max() <= 0.026


@pytest.mark.parametrize(
    ("angle_deg", "expected_deg"), [(0, 0), (91, -89), (-91, 89), (180, 0), (45, 45), (90, 90)]
)
def test_gripper_yaw_is_wrapped_to_half_a_turn(angle_deg: float, expected_deg: float) -> None:
    assert np.degrees(wrap_symmetric_yaw(np.radians(angle_deg))) == pytest.approx(expected_deg, abs=1e-6)


def test_retarget_plan_snaps_to_rim_and_centres_on_the_plate(setup: RealSetup) -> None:
    workspace = Workspace()
    mapping = table_to_world_map(setup, workspace)
    # The human recorded one layout; the sim layout is a different one.
    demo = synthetic_pick_place_demo(
        bowl_xy_table=np.array([0.6, -0.40]),
        plate_xy_table=np.array([0.9, -0.20]),
        start_xyz_table=np.array([0.1, -0.5, 0.3]),
        grasp_angle_rad=0.0,
        rim_radius_m=0.05,  # a sloppy grasp, 2 cm inside the rim
    )
    layout = SimLayout(bowl_xy=np.array([-0.10, 0.20]), plate_xy=np.array([0.05, -0.15]))
    plate_height = 0.031
    plan = ObjectCentricRetargeter(mapping, setup.bowl.radius_m, plate_height).plan(demo, layout)

    grasp_point = plan.approach_world[-1]
    assert np.linalg.norm(grasp_point[:2] - layout.bowl_xy) == pytest.approx(setup.bowl.radius_m, abs=1e-6)
    assert grasp_point[2] - SIM_TABLE_TOP_Z == pytest.approx(0.040, abs=1e-6)
    assert np.allclose(plan.bowl_path_world[0, :2], layout.bowl_xy, atol=1e-6)
    assert np.allclose(plan.bowl_path_world[-1, :2], layout.plate_xy, atol=1e-6)
    assert plan.bowl_path_world[-1, 2] == pytest.approx(SIM_TABLE_TOP_Z + plate_height + 0.004, abs=1e-6)
    # Only the last stretch of the approach is kept: the hand's far-away start is not.
    assert np.linalg.norm(plan.approach_world[0, :2] - grasp_point[:2]) <= 0.21
    # Pinch on the human's +x (table) side is the robot's -y side: radial along world y, so no yaw.
    assert abs(plan.gripper_yaw_rad) < np.radians(5)


def test_retarget_requires_grasp_and_release(setup: RealSetup) -> None:
    demo = synthetic_pick_place_demo(np.array([0.6, -0.4]), np.array([0.9, -0.2]), np.array([0.1, -0.5, 0.3]))
    demo.release_frame_idx = None
    retargeter = ObjectCentricRetargeter(table_to_world_map(setup, Workspace()), 0.069, 0.031)
    with pytest.raises(ValueError):
        retargeter.plan(demo, SimLayout(np.array([0.0, 0.1]), np.array([0.0, -0.1])))


def test_resample_timed_path_gives_each_point_its_time() -> None:
    from hurobo.retarget import resample_timed_path

    path = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    times = np.array([0.0, 1.0, 3.0, 4.0])
    points, point_times = resample_timed_path(path, times, 0.05)
    assert points.shape == (len(point_times), 3)
    assert point_times[0] == 0.0 and np.isclose(point_times[-1], 4.0)
    assert np.all(np.diff(point_times) >= 0)
    # the standstill (1 s to 3 s) falls between the points on either side of x = 0.1
    assert np.isclose(point_times[np.argmin(np.abs(points[:, 0] - 0.1))], 1.0, atol=1.0)
