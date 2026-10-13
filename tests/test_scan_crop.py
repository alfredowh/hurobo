"""Tests for the object crop: circle fit, RANSAC floor, floor + cluster estimate, white-object reference."""

from __future__ import annotations

import cv2
import numpy as np

from hurobo.scan import CropCylinder, PosedImage, SfmResult
from hurobo.scan_crop import (
    FloorPlane,
    WhiteObjectReference,
    choose_object_crop,
    crop_report,
    estimate_object_crop,
    extract_object_splat,
    fit_circle,
    fit_floor_plane,
    locate_white_object,
)


def _scene(rng: np.random.Generator, tilt_deg: float = 0.0):
    """Floor (z~0), a bowl-like cap at (0.3, -0.2), far clutter, and a camera ring of radius 4."""
    floor_xy = rng.uniform(-6, 6, size=(2000, 2))
    floor = np.c_[floor_xy, 0.01 * rng.normal(size=2000)]
    angle, azim = rng.uniform(0, np.pi / 2, 300), rng.uniform(0, 2 * np.pi, 300)
    bowl = np.c_[0.3 + np.sin(angle) * np.cos(azim), -0.2 + np.sin(angle) * np.sin(azim), np.cos(angle)]
    clutter = rng.uniform([15, -5, 1], [25, 5, 4], size=(100, 3))
    points = np.r_[floor, bowl, clutter]
    if tilt_deg:
        t = np.radians(tilt_deg)
        points = points @ np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]).T
    ring = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    cameras = np.c_[4 * np.cos(ring), 4 * np.sin(ring), np.full(60, 3.0)]
    return points, cameras


def test_fit_circle_recovers_centre_and_radius_from_a_partial_arc():
    angle = np.linspace(0.2, 2.2, 40)
    xy = np.c_[1.5 + 3.0 * np.cos(angle), -0.5 + 3.0 * np.sin(angle)]
    centre, radius = fit_circle(xy)
    np.testing.assert_allclose(centre, [1.5, -0.5], atol=1e-6)
    assert abs(radius - 3.0) < 1e-6


def test_fit_floor_plane_ignores_the_object_and_far_clutter():
    points, _ = _scene(np.random.default_rng(0))
    floor = fit_floor_plane(points, inlier_distance=0.05)
    assert floor is not None
    assert floor.tilt_deg < 1.0
    assert abs(floor.point[2]) < 0.02
    assert floor.noise_sigma < 0.05


def test_fit_floor_plane_returns_none_without_a_dominant_plane():
    sphere = np.random.default_rng(0).normal(size=(1500, 3))
    assert fit_floor_plane(sphere, inlier_distance=0.02) is None


def test_estimate_object_crop_finds_the_object_not_the_floor():
    points, cameras = _scene(np.random.default_rng(1))
    crop = estimate_object_crop(points, cameras)
    assert crop is not None
    np.testing.assert_allclose(crop.cylinder.centre_xy, [0.3, -0.2], atol=0.15)
    # The cap has radius 1; the crop keeps a modest rim, far below the floor's 6 or the ring's 4.
    assert 1.0 <= crop.cylinder.radius <= 1.8
    assert crop.cylinder.z_min < 0.1 and 0.9 < crop.cylinder.z_max < 2.0


def test_estimate_object_crop_handles_a_tilted_floor():
    points, cameras = _scene(np.random.default_rng(2), tilt_deg=5.0)
    t = np.radians(5.0)
    cameras = cameras @ np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]).T
    crop = estimate_object_crop(points, cameras)
    assert crop is not None
    assert abs(crop.floor.tilt_deg - 5.0) < 1.0


def test_estimate_object_crop_fits_the_floor_inside_the_camera_ring_only():
    # A room: most points are walls far outside the orbit, so the floor is under 30 % of all points
    # but dominant inside the ring.
    rng = np.random.default_rng(1)
    points, cameras = _scene(rng)
    walls = np.c_[
        rng.uniform(-30, 30, 8000), np.full(8000, 30.0) + rng.normal(size=8000), rng.uniform(0, 10, 8000)
    ]
    points = np.r_[points, walls]
    assert fit_floor_plane(points, inlier_distance=0.05) is None
    estimate = estimate_object_crop(points, cameras)
    assert estimate is not None
    assert estimate.floor.tilt_deg < 1.0


def test_estimate_object_crop_is_none_without_a_floor():
    rng = np.random.default_rng(3)
    blob = rng.normal(size=(1500, 3))
    ring = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    assert estimate_object_crop(blob, np.c_[4 * np.cos(ring), 4 * np.sin(ring), np.zeros(30)]) is None


def test_choose_object_crop_prefers_explicit_then_estimate_then_fallback():
    points, cameras = _scene(np.random.default_rng(4))
    explicit = CropCylinder(2.0, -1.0, 3.0, np.array([0.0, 0.0]))
    choice = choose_object_crop(points, cameras, explicit)
    assert choice.cylinder is explicit and choice.method == "explicit"
    assert choice.floor is not None  # the floor is still reported, to cut the object off it
    choice = choose_object_crop(points, cameras)
    assert choice.method == "floor + cluster" and choice.floor is not None
    blob = np.random.default_rng(5).normal(size=(1500, 3))
    choice = choose_object_crop(blob, cameras)
    assert choice.method == "percentile of all points" and choice.floor is None


def _white_disc_view(centre: np.ndarray, radius: float, eye: np.ndarray, focal: float, size: int = 240):
    forward = (centre - eye) / np.linalg.norm(centre - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    t_world_cam = np.eye(4)
    t_world_cam[:3, :3] = np.stack([right, down, forward], axis=1)
    t_world_cam[:3, 3] = eye
    k = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1.0]])
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = (150, 100, 50)  # brown floor
    pixel_radius = int(round(focal * radius / np.linalg.norm(centre - eye)))
    cv2.circle(image, (size // 2, size // 2), pixel_radius, (245, 245, 245), -1)
    return PosedImage(image.astype(np.float32) / 255.0, np.linalg.inv(t_world_cam), k)


def test_locate_white_object_triangulates_centre_and_radius():
    centre, radius = np.array([0.3, -0.2, 0.5]), 0.6
    eyes = [
        np.array([4 * np.cos(a), 4 * np.sin(a), 3.0]) + [0.3, -0.2, 0.0]
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ]
    images = [_white_disc_view(centre, radius, eye, focal=300.0) for eye in eyes]
    reference = locate_white_object(SfmResult(images, np.zeros((1, 3)), np.zeros((1, 3))))
    assert reference is not None and reference.num_views == 8
    np.testing.assert_allclose(reference.centre, centre, atol=0.05)
    assert abs(reference.radius - radius) / radius < 0.15


def test_locate_white_object_returns_none_without_enough_views():
    image = PosedImage(np.full((60, 60, 3), 0.3, np.float32), np.eye(4), np.eye(3))
    assert locate_white_object(SfmResult([image] * 3, np.zeros((1, 3)), np.zeros((1, 3)))) is None


def test_crop_report_scores_radius_offset_and_coverage(tmp_path):
    from plyfile import PlyData, PlyElement

    xyz = np.array([[0.0, 0.0, 0.5], [0.2, 0.0, 0.4], [5.0, 0.0, 0.1], [6.0, 1.0, 0.1]], np.float32)
    vertex = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(tmp_path / "s.ply"))

    reference = WhiteObjectReference(np.array([0.0, 0.0, 0.5]), 1.0, 8, 0.1)
    methods = {
        "tight": CropCylinder(1.5, -1.0, 2.0, np.array([0.1, 0.0])),
        "huge": CropCylinder(10.0, -1.0, 2.0, np.zeros(2)),
        "off": CropCylinder(1.5, -1.0, 2.0, np.array([2.0, 0.0])),
    }
    report = crop_report(methods, reference, tmp_path / "s.ply").set_index("method")
    assert report.loc["tight", "object inside"] and report.loc["huge", "object inside"]
    assert not report.loc["off", "object inside"]
    assert report.loc["tight", "radius / object radius"] == 1.5
    assert report.loc["tight", "Gaussians kept (%)"] == 50.0
    assert report.loc["huge", "Gaussians kept (%)"] == 100.0


def _write_splat(path, xyz, opacity):
    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    names += ["f_dc_0", "f_dc_1", "f_dc_2"]
    from plyfile import PlyData, PlyElement

    vertex = np.zeros(len(xyz), dtype=[(n, "f4") for n in names])
    vertex["x"], vertex["y"], vertex["z"] = np.asarray(xyz, np.float32).T
    vertex["opacity"] = np.log(opacity / (1 - opacity))
    for axis in range(3):
        vertex[f"scale_{axis}"] = np.log(0.03)
    vertex["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


def test_extract_object_splat_keeps_the_translucent_bowl_and_frames_it_upright(tmp_path):
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    floor_xy = rng.uniform(-6, 6, size=(3000, 2))
    floor = np.c_[floor_xy, 0.02 * rng.normal(size=3000)]
    angle, azim = rng.uniform(0, np.pi / 2, 600), rng.uniform(0, 2 * np.pi, 600)
    bowl = np.c_[0.3 + np.sin(angle) * np.cos(azim), -0.2 + np.sin(angle) * np.sin(azim), np.cos(angle)]
    floaters = rng.uniform([4, 4, 2], [6, 6, 3], size=(20, 3))
    haze = np.c_[rng.uniform(-3, 3, size=(50, 2)), rng.uniform(-1.5, -0.6, size=50)]
    tilt = Rotation.from_euler("x", 3.0, degrees=True)
    points = tilt.apply(np.r_[floor, bowl, floaters, haze])
    opacity = np.r_[np.full(3000, 0.9), np.full(600, 0.06), np.full(20, 0.9), np.full(50, 0.5)]
    _write_splat(tmp_path / "scene.ply", points, opacity)

    floor_plane = FloorPlane(tilt.apply([0, 0, 1.0]), np.zeros(3), 0.5, 0.02)
    cylinder = CropCylinder(2.0, -1.0, 3.0, tilt.apply([0.3, -0.2, 0.0])[:2])
    info = extract_object_splat(tmp_path / "scene.ply", tmp_path / "bowl.ply", floor_plane, cylinder)

    out = PlyData.read(str(tmp_path / "bowl.ply"))["vertex"].data
    xyz = np.stack([out["x"], out["y"], out["z"]], axis=1)
    assert 500 <= info.num_gaussians <= 600  # the translucent bowl survives; floor, floaters and haze do not
    assert xyz[:, 2].min() > -0.05 and 0.9 < xyz[:, 2].max() < 1.1  # floor level is z = 0, bowl 1 high
    assert np.linalg.norm(np.median(xyz[:, :2], axis=0)) < 0.15  # centred on the axis
    assert np.linalg.norm(xyz[:, :2], axis=1).max() < 1.3  # no floaters
    assert abs(info.tilt_removed_deg - 3.0) < 0.1
