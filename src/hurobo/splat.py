"""Photoreal real2sim render: Gaussian splats attached to MuJoCo bodies, depth-composited with the mesh robot.

Recipe (REAL2SIM_IDEAS.md §7): splats replace the *appearance* of chosen sim
bodies (table, bowl, plate); physics, collision and the robot stay LIBERO's own.
The robot is never splatted (there's no real arm to scan) — it's the MuJoCo mesh,
merged per pixel by depth against the splat render (Re3Sim's hybrid z-buffer).
The `.ply` loader is DISCOVERSE's `gaussian_renderer.load_ply`, reused as-is; the
rasterization call is our own so it can return alpha + expected depth, which
DISCOVERSE's `batch_render` drops and which the z-test needs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hurobo.scan import SfmResult, Similarity
from hurobo.utils.transforms import look_at_t_cam_world

#: robosuite puts visual geoms in group 1 and collision geoms in group 0; only
#: group 1 is drawn in the mesh layer, so collision proxies never leak into frames.
_VISUAL_GEOM_GROUP = 1
#: A group the mesh layer never draws; replaced/hidden bodies' visual geoms are
#: parked here for the duration of one render, then restored.
_HIDDEN_GEOM_GROUP = 5
#: Below this splat opacity a pixel counts as "no splat" for the z-test, since
#: expected depth is ill-defined where almost nothing was accumulated.
_MIN_SPLAT_ALPHA = 0.05


class SplatFit(Enum):
    """How a splat's own frame is mapped into its MuJoCo body's frame.

    All modes centre the splat in x/y on the body's visual bounding box.
    METRIC_TOP: keep metric scale, put the splat's top on the box's top (the
    table, whose top surface is what objects rest on).
    METRIC_BOTTOM: keep metric scale, sit the splat's base on the box's bottom
    (a ChArUco-scaled scan of a real object).
    MATCH_BODY_XY: uniformly rescale so the splat's x/y footprint matches the sim
    asset's, then sit its base on the box's bottom. For splats with arbitrary
    scale (no ChArUco) or of a different physical object, so the visual and the
    collision geometry don't disagree.
    """

    METRIC_TOP = "metric_top"
    METRIC_BOTTOM = "metric_bottom"
    MATCH_BODY_XY = "match_body_xy"


@dataclass
class SplatAsset:
    """One splat `.ply` attached rigidly to one MuJoCo body.

    Attributes:
        body_name: MuJoCo body whose appearance the splat replaces, e.g. "plate_1_main".
        ply_path: 3DGS `.ply`, z-up, in the object's own frame.
        fit: how the splat is placed in the body frame (see `SplatFit`).
        yaw_deg: extra rotation about the splat's own z axis before fitting.
    """

    body_name: str
    ply_path: Path
    fit: SplatFit = SplatFit.METRIC_BOTTOM
    yaw_deg: float = 0.0


#: Where the DISCOVERSE stand-in splats are cached, and their paths in the HF repo `tatp/DISCOVERSE-models`.
DUMMY_SPLAT_DIR = Path("data/splats/dummy_discoverse")
DUMMY_HF_FILES = {
    "object__table_cloth_0.ply": "3dgs/object/table_cloth_0.ply",
    "object__bowl_pink.ply": "3dgs/object/bowl_pink.ply",
    "object__plate_white.ply": "3dgs/object/plate_white.ply",
}

#: libero_goal objects that aren't on the real table; hidden from the composite.
LIBERO_GOAL_DISTRACTOR_ROOTS = (
    "cream_cheese_1_main",
    "wine_bottle_1_main",
    "wooden_cabinet_1_main",
    "flat_stove_1_main",
    "wine_rack_1_main",
)


def build_assets(
    bowl_ply: Path | None = None,
    plate_ply: Path | None = None,
    table_ply: Path | None = None,
    bowl_fit: SplatFit = SplatFit.MATCH_BODY_XY,
    bowl_yaw_deg: float = 0.0,
) -> list[SplatAsset]:
    """Splats for table, bowl and plate: given .ply files, else DISCOVERSE stand-ins (downloaded if missing).

    Bowl/plate use MATCH_BODY_XY since neither the stand-ins nor a ChArUco-less
    scan (`09_scan_object.py`) are at the sim asset's scale.

    Args:
        bowl_ply: splat for `akita_black_bowl_1_main`; default the DISCOVERSE pink bowl.
        plate_ply: splat for `plate_1_main`; default the DISCOVERSE white plate.
        table_ply: splat for the table; default the DISCOVERSE table cloth.
        bowl_fit: how the bowl splat is fitted to the sim body.
        bowl_yaw_deg: extra yaw of the bowl splat before fitting.
    """
    from huggingface_hub import hf_hub_download

    DUMMY_SPLAT_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, repo_path in DUMMY_HF_FILES.items():
        local_path = DUMMY_SPLAT_DIR / local_name
        if not local_path.exists():
            local_path.write_bytes(Path(hf_hub_download("tatp/DISCOVERSE-models", repo_path)).read_bytes())
    # The cloth's long side is its x; the LIBERO table's long side is world y.
    table = (
        SplatAsset("table", table_ply, SplatFit.METRIC_TOP)
        if table_ply
        else SplatAsset(
            "table", DUMMY_SPLAT_DIR / "object__table_cloth_0.ply", SplatFit.METRIC_TOP, yaw_deg=90.0
        )
    )
    return [
        table,
        SplatAsset(
            "akita_black_bowl_1_main",
            bowl_ply or DUMMY_SPLAT_DIR / "object__bowl_pink.ply",
            bowl_fit,
            yaw_deg=bowl_yaw_deg,
        ),
        SplatAsset(
            "plate_1_main", plate_ply or DUMMY_SPLAT_DIR / "object__plate_white.ply", SplatFit.MATCH_BODY_XY
        ),
    ]


@dataclass
class CompositeFrame:
    """One composited render plus its layers, all shape (H, W, 3) uint8 RGB except masks.

    Attributes:
        composite: splats and mesh merged by depth — the frame a policy would see.
        splat_rgb: splat layer alone, over black.
        mesh_rgb: mesh layer alone (robot + any body not replaced or hidden).
        splat_wins: shape (H, W) bool, pixels where the splat was in front of the mesh.
    """

    composite: np.ndarray
    splat_rgb: np.ndarray
    mesh_rgb: np.ndarray
    splat_wins: np.ndarray


def body_visual_aabb(
    model: mujoco.MjModel, body_id: int, geom_group: int = _VISUAL_GEOM_GROUP
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of one body's geoms of one group (default: visual), in the body frame.

    Args:
        model: the raw MuJoCo model.
        body_id: id of the body.
        geom_group: 1 = visual, 0 = collision in robosuite.

    Returns:
        (lo_body, hi_body), each shape (3,).

    Raises:
        ValueError: if the body has no geoms in that group.
    """
    corners_body = []
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
        if model.geom_group[geom_id] != geom_group:
            continue
        # geom_aabb is (centre, half-size) in the geom's own frame.
        centre_geom, half_geom = model.geom_aabb[geom_id, :3], model.geom_aabb[geom_id, 3:]
        corners_geom = centre_geom + signs * half_geom
        rot_body_geom = Rotation.from_quat(model.geom_quat[geom_id], scalar_first=True)
        corners_body.append(rot_body_geom.apply(corners_geom) + model.geom_pos[geom_id])
    if not corners_body:
        raise ValueError(f"body {body_id} has no group-{geom_group} geoms")
    all_corners = np.concatenate(corners_body)
    return all_corners.min(axis=0), all_corners.max(axis=0)


def fit_gaussians_to_body(
    gaussians, asset: SplatAsset, body_lo: np.ndarray, body_hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a splat from its own frame into its body's frame, per `asset.fit`.

    Shared by `SplatCompositor` (what is drawn) and `collider.py` (what is
    collided with), so the visible splat and its collider cannot drift apart.

    Args:
        gaussians: a `gaussian_renderer` GaussianData (xyz, rot wxyz, scale, opacity, sh).
        asset: which splat and how to fit it.
        body_lo: shape (3,), lower corner of the sim body's visual AABB, body frame.
        body_hi: shape (3,), upper corner.

    Returns:
        (xyz_body, rot_body_wxyz, scale), numpy float32.
    """
    rot_yaw = Rotation.from_euler("z", asset.yaw_deg, degrees=True)
    xyz = rot_yaw.apply(gaussians.xyz)
    rot = (rot_yaw * Rotation.from_quat(gaussians.rot, scalar_first=True)).as_quat(scalar_first=True)
    scale = np.asarray(gaussians.scale, dtype=np.float64)

    # Percentiles of the opaque Gaussians, not min/max: scans carry a few floaters.
    solid = xyz[gaussians.opacity > 0.3]
    splat_lo, splat_hi = np.percentile(solid, 2, axis=0), np.percentile(solid, 98, axis=0)

    if asset.fit is SplatFit.MATCH_BODY_XY:
        factor = float(np.mean((body_hi - body_lo)[:2] / (splat_hi - splat_lo)[:2]))
        xyz, scale = xyz * factor, scale * factor
        splat_lo, splat_hi = splat_lo * factor, splat_hi * factor
    if asset.fit is SplatFit.METRIC_TOP:
        z_offset = body_hi[2] - splat_hi[2]
    else:
        z_offset = body_lo[2] - splat_lo[2]

    xy_offset = (body_lo + body_hi)[:2] / 2 - (splat_lo + splat_hi)[:2] / 2
    xyz = xyz + np.array([xy_offset[0], xy_offset[1], z_offset])
    return xyz.astype(np.float32), rot.astype(np.float32), scale.astype(np.float32)


def splat_centre_and_radius(xyz: np.ndarray, opacity: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust centre and radius of a splat's solid part (floaters are excluded by opacity and percentiles).

    Args:
        xyz: shape (N, 3) Gaussian centres.
        opacity: shape (N,) in [0, 1].

    Returns:
        (centre of the 2-98 percentile box, distance from it to that box's upper corner).
    """
    solid = xyz[opacity > 0.3]
    lo, hi = np.percentile(solid, 2, axis=0), np.percentile(solid, 98, axis=0)
    centre = (lo + hi) / 2
    return centre, float(np.linalg.norm(hi - centre))


def splat_dimensions(ply_path: Path) -> tuple[float, float]:
    """Footprint diameter and height of a splat's solid part, in the splat's own units.

    Uses the same 2-98 percentile box of opaque Gaussians as `fit_gaussians_to_body`, so the number is
    what `MATCH_BODY_XY` would rescale.

    Args:
        ply_path: object splat, z-up, base at z = 0.

    Returns:
        (mean of the x and y extents, z extent). For a scan these are SfM units, not metres, unless the
        scan was made with a ChArUco board.
    """
    from gaussian_renderer import load_ply

    gaussians = load_ply(str(ply_path))
    solid = gaussians.xyz[gaussians.opacity > 0.3]
    extent = np.percentile(solid, 98, axis=0) - np.percentile(solid, 2, axis=0)
    return float(extent[:2].mean()), float(extent[2])


@torch.no_grad()
def render_splat_orbit(
    ply_path: Path,
    num_views: int = 12,
    image_size: int = 256,
    elevation_deg: float = 30.0,
    background: float = 0.92,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Rasterise a splat (full anisotropic Gaussians, opacity and colour) from cameras orbiting its centre.

    This is the splat as the compositor would draw it, not just its centres, so
    the surface shape of the object is visible.

    Args:
        ply_path: 3DGS `.ply`, z-up, in the splat's own frame.
        num_views: camera positions, evenly spaced in azimuth.
        image_size: square output size in pixels.
        elevation_deg: camera elevation above the splat's xy plane.
        background: grey level in [0, 1] composited behind the splat.
        device: torch device for rasterisation.

    Returns:
        `num_views` images, each (image_size, image_size, 3) uint8 RGB.
    """
    from gaussian_renderer import load_ply
    from gsplat.rendering import rasterization

    gaussians = load_ply(str(ply_path))
    centre, radius = splat_centre_and_radius(gaussians.xyz, gaussians.opacity)
    fov_deg = 40.0
    focal = (image_size / 2) / np.tan(np.radians(fov_deg / 2))
    distance = 1.3 * radius / np.tan(np.radians(fov_deg / 2))
    elevation = np.radians(elevation_deg)

    def to_dev(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).float().to(device)

    k = to_dev(np.array([[focal, 0, image_size / 2], [0, focal, image_size / 2], [0, 0, 1.0]]))[None]
    means, quats, scales = to_dev(gaussians.xyz), to_dev(gaussians.rot), to_dev(gaussians.scale)
    opacities, colors = to_dev(gaussians.opacity), to_dev(gaussians.sh[:, :3])[:, None, :]

    images = []
    for azimuth in np.linspace(0, 2 * np.pi, num_views, endpoint=False):
        eye_world = centre + distance * np.array(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
        )
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=to_dev(look_at_t_cam_world(eye_world, centre))[None],
            Ks=k,
            width=image_size,
            height=image_size,
            sh_degree=0,
        )
        # Premultiplied colour over a flat background.
        over = render[0] + (1.0 - alpha[0]) * background
        images.append((over.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return images


@torch.no_grad()
def render_splat_views(
    ply_path: Path,
    t_cam_world_list: list[np.ndarray],
    ks: list[np.ndarray],
    width: int,
    height: int,
    background: float = 0.0,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Rasterise a splat from given cameras, e.g. the phone's own path through the scene.

    Args:
        ply_path: 3DGS `.ply`.
        t_cam_world_list: 4x4 world -> OpenCV camera transforms, in the splat's frame.
        ks: 3x3 pinhole intrinsics, one per camera, at (width, height).
        width: image width in pixels.
        height: image height in pixels.
        background: grey level in [0, 1] behind the splat (the scan was trained on black).
        device: torch device.

    Returns:
        One (height, width, 3) uint8 RGB image per camera.
    """
    from gaussian_renderer import load_ply
    from gsplat.rendering import rasterization

    gaussians = load_ply(str(ply_path))

    def to_dev(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).float().to(device)

    means, quats, scales = to_dev(gaussians.xyz), to_dev(gaussians.rot), to_dev(gaussians.scale)
    opacities, colors = to_dev(gaussians.opacity), to_dev(gaussians.sh[:, :3])[:, None, :]
    images = []
    for t_cam_world, k in zip(t_cam_world_list, ks, strict=True):
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=to_dev(t_cam_world)[None],
            Ks=to_dev(k)[None],
            width=width,
            height=height,
            sh_degree=0,
        )
        over = render[0] + (1.0 - alpha[0]) * background
        images.append((over.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return images


def pinhole_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """3x3 pinhole intrinsics with the principal point at the image centre.

    Args:
        width: image width in pixels.
        height: image height in pixels.
        fov_deg: horizontal field of view in degrees.

    Returns:
        (3, 3) float camera matrix.
    """
    focal = (width / 2) / np.tan(np.radians(fov_deg / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def arc_eyes(
    target_world: np.ndarray, radius: float, height: float, azimuth_range_deg: tuple[float, float], num: int
) -> list[np.ndarray]:
    """Camera positions on a horizontal arc around a point, for a sweep that stays on the scanned side.

    Args:
        target_world: arc centre (the point the cameras look at).
        radius: horizontal distance from the centre.
        height: camera height above the centre.
        azimuth_range_deg: first and last azimuth, measured from +x towards +y.
        num: number of positions.

    Returns:
        `num` positions in the target's frame.
    """
    azimuths = np.radians(np.linspace(*azimuth_range_deg, num))
    offsets = np.stack([radius * np.cos(azimuths), radius * np.sin(azimuths), np.full(num, height)], axis=1)
    return [np.asarray(target_world) + offset for offset in offsets]


def render_scene_views(
    ply_path: Path,
    eyes_world: list[np.ndarray],
    target_world: np.ndarray,
    width: int = 960,
    height: int = 600,
    fov_deg: float = 55.0,
) -> list[np.ndarray]:
    """Render a whole-scene splat from chosen viewpoints looking at one point, like the browser viewer.

    Args:
        ply_path: scene splat, z up (the table frame for the scans).
        eyes_world: camera positions in the splat's frame.
        target_world: the point every camera looks at.
        width: image width in pixels.
        height: image height in pixels.
        fov_deg: horizontal field of view in degrees.

    Returns:
        One (height, width, 3) uint8 RGB image per eye.
    """
    k = pinhole_intrinsics(width, height, fov_deg)
    return render_splat_views(
        ply_path,
        [look_at_t_cam_world(np.asarray(eye), np.asarray(target_world)) for eye in eyes_world],
        [k] * len(eyes_world),
        width,
        height,
    )


def render_scan_path(
    ply_path: Path, sfm: SfmResult, similarity: Similarity, num_views: int = 12
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """Render a scan's splat from evenly spaced keyframe cameras, next to the photos they were taken from.

    Args:
        ply_path: the splat, already moved into the similarity's destination frame (`transform_ply`).
        sfm: the scan's posed keyframes (SfM frame).
        similarity: SfM frame -> the splat's frame.
        num_views: keyframes to render, spread over the whole capture path.

    Returns:
        (keyframe indices, photos, renders), photos and renders as (H, W, 3) uint8 RGB.
    """
    t_cam_world = sfm.t_cam_world_in(similarity)
    ids = sorted({int(i) for i in np.linspace(0, len(sfm.images) - 1, num_views).round()})
    height, width = sfm.images[0].image_rgb.shape[:2]
    renders = render_splat_views(
        ply_path, [t_cam_world[i] for i in ids], [sfm.images[i].k for i in ids], width, height
    )
    photos = [(sfm.images[i].image_rgb * 255).astype(np.uint8) for i in ids]
    return ids, photos, renders


def render_asset_composites(
    asset_sets: dict[str, list[SplatAsset]],
    size: int = 384,
    camera_names: tuple[str, ...] = ("agentview",),
    suite: str = "libero_goal",
    task_id: int = 8,
) -> dict[str, np.ndarray]:
    """Each asset set composited into LIBERO's initial scene; one image per set, cameras side by side.

    Args:
        asset_sets: label -> splats to attach (see `build_assets`).
        size: square size of one camera image in pixels.
        camera_names: LIBERO cameras to render, joined left to right.
        suite: LIBERO suite name.
        task_id: task index in the suite.

    Returns:
        label -> (size, size * len(camera_names), 3) uint8 RGB depth composite (splats + mesh robot).
    """
    from hurobo.sim import LiberoTask

    task = LiberoTask(suite, task_id)
    task.reset()
    images = {}
    try:
        for label, assets in asset_sets.items():
            compositor = SplatCompositor(
                task.mj_model,
                assets,
                hidden_body_roots=LIBERO_GOAL_DISTRACTOR_ROOTS,
                width=size,
                height=size,
            )
            try:
                views = [compositor.render_layers(task.mj_data, name)["composite"] for name in camera_names]
                images[label] = np.concatenate(views, axis=1)
            finally:
                compositor.close()
    finally:
        task.close()
    return images


def render_bowl_composites(
    bowl_plys: dict[str, Path],
    size: int = 384,
    camera_name: str = "agentview",
    suite: str = "libero_goal",
    task_id: int = 8,
    bowl_fit: SplatFit = SplatFit.MATCH_BODY_XY,
) -> dict[str, np.ndarray]:
    """Each bowl splat attached in turn to LIBERO's bowl body; one composited camera image per splat.

    The table and plate use the DISCOVERSE stand-ins, so only the bowl differs between images.

    Args:
        bowl_plys: label -> object splat (z-up, base at z = 0), e.g. from `extract_object_splat`.
        size: square image size in pixels.
        camera_name: LIBERO camera to render.
        suite: LIBERO suite name.
        task_id: task index in the suite.
        bowl_fit: how each splat is fitted to the body.

    Returns:
        label -> (size, size, 3) uint8 RGB depth composite (splats + mesh robot).
    """
    return render_asset_composites(
        {label: build_assets(bowl_ply=ply, bowl_fit=bowl_fit) for label, ply in bowl_plys.items()},
        size=size,
        camera_names=(camera_name,),
        suite=suite,
        task_id=task_id,
    )


def _quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (…, 4) scalar-first quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


class SplatCompositor:
    """Renders a MuJoCo scene with some bodies' appearance replaced by Gaussian splats.

    One instance per model; call `render` with the current `MjData` each frame.
    Splats follow their bodies' `xpos`/`xquat`, so a grasped, moving bowl moves
    its splat with it.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        assets: list[SplatAsset],
        hidden_body_roots: tuple[str, ...] = (),
        width: int = 512,
        height: int = 512,
        device: str = "cuda",
    ) -> None:
        """Load, fit and upload every splat; set up the mesh renderer.

        Args:
            model: the raw MuJoCo model (robosuite: `sim.model._model`).
            assets: splats to attach; each replaces its body's visual geoms.
            hidden_body_roots: root body names whose whole subtree is removed from
                the mesh layer without a splat replacement (LIBERO distractors
                that don't exist in the real scene).
            width: output width in pixels.
            height: output height in pixels.
            device: torch device for rasterization.
        """
        from gaussian_renderer import load_ply

        self._model = model
        self._width, self._height = width, height
        self._device = device

        xyz, rot, scale, opacity, sh, owner = [], [], [], [], [], []
        self._body_ids = []
        for slot, asset in enumerate(assets):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, asset.body_name)
            if body_id < 0:
                raise ValueError(f"no body named {asset.body_name!r} in the model")
            gaussians = load_ply(str(asset.ply_path))
            body_lo, body_hi = body_visual_aabb(model, body_id)
            g_xyz, g_rot, g_scale = fit_gaussians_to_body(gaussians, asset, body_lo, body_hi)
            xyz.append(g_xyz)
            rot.append(g_rot)
            scale.append(g_scale)
            opacity.append(gaussians.opacity)
            sh.append(gaussians.sh[:, :3])  # DC term only: stand-in splats are degree 0 anyway
            owner.append(np.full(len(g_xyz), slot))
            self._body_ids.append(body_id)

        def to_dev(arrays: list[np.ndarray]) -> torch.Tensor:
            return torch.from_numpy(np.concatenate(arrays)).float().to(device)

        self._xyz_body = to_dev(xyz)
        self._rot_body = to_dev(rot)
        self._scale = to_dev(scale)
        self._opacity = to_dev(opacity)
        self._sh = to_dev(sh).reshape(-1, 1, 3)
        self._owner = torch.from_numpy(np.concatenate(owner)).long().to(device)

        hidden_ids = set(self._body_ids)
        for root_name in hidden_body_roots:
            root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
            if root_id < 0:
                raise ValueError(f"no body named {root_name!r} in the model")
            hidden_ids.update(np.flatnonzero(model.body_rootid == root_id).tolist())
        self._hidden_geom_ids = np.flatnonzero(
            np.isin(model.geom_bodyid, list(hidden_ids)) & (model.geom_group == _VISUAL_GEOM_GROUP)
        )

        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self._mesh_renderer = mujoco.Renderer(model, height=height, width=width)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        self._scene_option.geomgroup[_VISUAL_GEOM_GROUP] = 1
        # Sites (LIBERO region markers, the gripper's grip-site cylinder) are drawn by
        # default and would win the z-test against the splat table; policy cameras never show them.
        self._scene_option.sitegroup[:] = 0

    @property
    def num_gaussians(self) -> int:
        """Total number of Gaussians across all attached splats."""
        return int(self._xyz_body.shape[0])

    def _posed_gaussians(self, data: mujoco.MjData) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame Gaussian means and orientations for the current body poses."""
        p_world_body = torch.from_numpy(data.xpos[self._body_ids]).float().to(self._device)
        q_world_body = torch.from_numpy(data.xquat[self._body_ids]).float().to(self._device)
        rot_world_body = (
            torch.from_numpy(data.xmat[self._body_ids].reshape(-1, 3, 3)).float().to(self._device)
        )
        r = rot_world_body[self._owner]
        xyz_world = torch.einsum("nij,nj->ni", r, self._xyz_body) + p_world_body[self._owner]
        rot_world = _quat_multiply_wxyz(q_world_body[self._owner], self._rot_body)
        return xyz_world, rot_world

    def _camera(self, data: mujoco.MjData, camera_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """gsplat view matrix (world->OpenCV camera) and intrinsics for a MuJoCo camera."""
        # MuJoCo cameras look down their -z with +y up; OpenCV looks down +z with +y down.
        rot_world_cam_mj = data.cam_xmat[camera_id].reshape(3, 3)
        t_world_cam_cv = np.eye(4)
        t_world_cam_cv[:3, :3] = rot_world_cam_mj @ np.diag([1.0, -1.0, -1.0])
        t_world_cam_cv[:3, 3] = data.cam_xpos[camera_id]
        t_cam_world = np.linalg.inv(t_world_cam_cv)

        # MuJoCo renders with square pixels and a principal point at the centre.
        fy = self._height / (2.0 * np.tan(np.radians(self._model.cam_fovy[camera_id]) / 2.0))
        k = np.array([[fy, 0, self._width / 2], [0, fy, self._height / 2], [0, 0, 1]])
        to_dev = lambda a: torch.from_numpy(a).float().to(self._device)[None]  # noqa: E731
        return to_dev(t_cam_world), to_dev(k)

    @torch.no_grad()
    def _render_splats(self, data: mujoco.MjData, camera_id: int) -> tuple[np.ndarray, ...]:
        """Rasterize the splats: (premultiplied rgb float, alpha, expected depth in m)."""
        from gsplat.rendering import rasterization

        xyz_world, rot_world = self._posed_gaussians(data)
        viewmat, k = self._camera(data, camera_id)
        renders, alphas, _meta = rasterization(
            means=xyz_world,
            quats=rot_world,
            scales=self._scale,
            opacities=self._opacity,
            colors=self._sh,
            viewmats=viewmat,
            Ks=k,
            width=self._width,
            height=self._height,
            sh_degree=0,
            render_mode="RGB+ED",
        )
        rgb = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
        depth = renders[0, ..., 3].cpu().numpy()
        alpha = alphas[0, ..., 0].cpu().numpy()
        return rgb, alpha, depth

    @contextmanager
    def _replaced_geoms_hidden(self) -> Iterator[None]:
        """Temporarily move replaced/hidden bodies' visual geoms out of the drawn group."""
        original_groups = self._model.geom_group[self._hidden_geom_ids].copy()
        self._model.geom_group[self._hidden_geom_ids] = _HIDDEN_GEOM_GROUP
        try:
            yield
        finally:
            self._model.geom_group[self._hidden_geom_ids] = original_groups

    def _render_mesh(self, data: mujoco.MjData, camera_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Mesh layer: (rgb uint8, metric depth) without the replaced/hidden bodies."""
        with self._replaced_geoms_hidden():
            self._mesh_renderer.update_scene(data, camera=camera_id, scene_option=self._scene_option)
            rgb = self._mesh_renderer.render().copy()
            self._mesh_renderer.enable_depth_rendering()
            self._mesh_renderer.update_scene(data, camera=camera_id, scene_option=self._scene_option)
            depth = self._mesh_renderer.render().copy()
            self._mesh_renderer.disable_depth_rendering()
        return rgb, depth

    def render(self, data: mujoco.MjData, camera_name: str) -> CompositeFrame:
        """Render one frame: splats and mesh merged by per-pixel depth.

        Args:
            data: the raw MuJoCo data, already `mj_forward`-ed (robosuite: `sim.data._data`).
            camera_name: a camera defined in the model, e.g. "agentview".

        Returns:
            The composite and its layers.
        """
        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"no camera named {camera_name!r} in the model")
        splat_rgb, splat_alpha, splat_depth = self._render_splats(data, camera_id)
        mesh_rgb, mesh_depth = self._render_mesh(data, camera_id)

        mesh_rgb_f = mesh_rgb.astype(np.float32) / 255.0
        splat_wins = (splat_alpha > _MIN_SPLAT_ALPHA) & (splat_depth < mesh_depth)
        # splat_rgb is premultiplied by alpha (gsplat accumulates alpha-weighted colour).
        over = splat_rgb + (1.0 - splat_alpha[..., None]) * mesh_rgb_f
        composite = np.where(splat_wins[..., None], over, mesh_rgb_f)

        def to_u8(image: np.ndarray) -> np.ndarray:
            return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

        return CompositeFrame(
            composite=to_u8(composite), splat_rgb=to_u8(splat_rgb), mesh_rgb=mesh_rgb, splat_wins=splat_wins
        )

    def render_layers(self, data: mujoco.MjData, camera_name: str) -> dict[str, np.ndarray]:
        """Native LIBERO render beside the composite's layers, as named RGB panels for plotting.

        Args:
            data: the raw MuJoCo data; `mj_forward` is called first so it reflects moved bodies.
            camera_name: a camera defined in the model.

        Returns:
            Ordered {"LIBERO native", "mesh layer", "splat layer", "composite"} -> (H, W, 3) uint8.
        """
        mujoco.mj_forward(self._model, data)
        frame = self.render(data, camera_name)
        return {
            "LIBERO native": self.render_native(data, camera_name),
            "mesh layer": frame.mesh_rgb,
            "splat layer": frame.splat_rgb,
            "composite": frame.composite,
        }

    def render_collision_mask(self, data: mujoco.MjData, camera_name: str, body_name: str) -> np.ndarray:
        """Boolean (H, W) mask of one body's collision geoms (group 0) as seen from a camera.

        Used to overlay the physics shape on the splat render, to check they coincide.
        Occlusion by other geoms is respected (a segmentation render), so a gripper
        in front of the collider hides it.
        """
        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[0] = 1
        option.sitegroup[:] = 0
        self._mesh_renderer.enable_segmentation_rendering()
        self._mesh_renderer.update_scene(data, camera=camera_id, scene_option=option)
        segmentation = self._mesh_renderer.render().copy()
        self._mesh_renderer.disable_segmentation_rendering()
        is_geom = segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        geom_ids = np.flatnonzero(self._model.geom_bodyid == body_id)
        return is_geom & np.isin(segmentation[..., 0], geom_ids)

    def render_native(self, data: mujoco.MjData, camera_name: str) -> np.ndarray:
        """Plain LIBERO render (all visual geoms, no splats) through the same renderer.

        Same renderer and image orientation as `render`, so the two are directly
        comparable side by side (robosuite's own `sim.render` is vertically flipped).
        """
        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self._mesh_renderer.update_scene(data, camera=camera_id, scene_option=self._scene_option)
        return self._mesh_renderer.render().copy()

    def close(self) -> None:
        """Release the mesh renderer's GL context."""
        self._mesh_renderer.close()
