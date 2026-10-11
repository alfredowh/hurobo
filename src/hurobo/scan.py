"""Orbit video -> posed keyframes (pycolmap SfM) -> trained Gaussian splat (gsplat) -> object-frame .ply.

The capture half of the photoreal stretch goal (REAL2SIM_IDEAS.md §7, PolaRiS's
recipe on DISCOVERSE's renderer). Output `.ply` files are standard 3DGS format,
read by `splat.SplatCompositor` via DISCOVERSE's `load_ply`.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hurobo.utils.video import read_frames


def sharpness(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian: higher means less motion blur."""
    return float(cv2.Laplacian(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


class KeyframeSampler:
    """Picks the sharpest frame in each of N equal time bins of an orbit video."""

    def __init__(self, num_frames: int = 120, max_side_px: int = 960) -> None:
        """Set how many keyframes to keep and the size they're saved at.

        Args:
            num_frames: number of keyframes (time bins) to keep.
            max_side_px: longer image side after downscaling; SfM and training
                both run at this size.
        """
        self._num_frames = num_frames
        self._max_side_px = max_side_px

    def sample(self, video_path: Path, out_dir: Path) -> list[Path]:
        """Write the keyframes as PNGs and return their paths, in time order.

        Args:
            video_path: the orbit video.
            out_dir: emptied and refilled with `frame_XXXX.png` files.

        Returns:
            Paths of the written keyframes.
        """
        frames = list(read_frames(video_path))
        bins = np.array_split(np.arange(len(frames)), min(self._num_frames, len(frames)))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        paths = []
        for i, frame_ids in enumerate(bins):
            best = max(frame_ids, key=lambda idx: sharpness(frames[idx]))
            image = frames[best]
            scale = self._max_side_px / max(image.shape[:2])
            if scale < 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            path = out_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(path), image)
            paths.append(path)
        return paths


@dataclass
class PosedImage:
    """One SfM-registered keyframe, as the splat trainer needs it.

    Attributes:
        image_rgb: shape (H, W, 3) float32 in [0, 1].
        t_cam_world: 4x4, world (SfM) frame -> OpenCV camera frame.
        k: 3x3 pinhole intrinsics at this image's resolution.
    """

    image_rgb: np.ndarray
    t_cam_world: np.ndarray
    k: np.ndarray


@dataclass
class SfmResult:
    """Posed keyframes plus the sparse point cloud, all in the SfM frame.

    Attributes:
        images: the registered keyframes.
        points_world: shape (P, 3) sparse SfM points.
        colors: shape (P, 3) their RGB colours in [0, 1].
    """

    images: list[PosedImage]
    points_world: np.ndarray
    colors: np.ndarray

    @property
    def camera_centres_world(self) -> np.ndarray:
        """Shape (N, 3) camera centres in the SfM frame."""
        return np.array([np.linalg.inv(im.t_cam_world)[:3, 3] for im in self.images])

    @property
    def t_world_cam_list(self) -> list[np.ndarray]:
        """4x4 camera-to-world transforms of the keyframes (OpenCV camera axes)."""
        return [np.linalg.inv(im.t_cam_world) for im in self.images]

    def in_frame(self, t_new_world: np.ndarray) -> SfmResult:
        """The same reconstruction re-expressed in another frame (points and cameras move, pixels don't).

        Args:
            t_new_world: 4x4, current SfM frame -> new frame, e.g. `object_frame_from_orbit(sfm)`.
        """
        t_world_new = np.linalg.inv(t_new_world)
        images = [PosedImage(im.image_rgb, im.t_cam_world @ t_world_new, im.k) for im in self.images]
        points = self.points_world @ t_new_world[:3, :3].T + t_new_world[:3, 3]
        return SfmResult(images=images, points_world=points, colors=self.colors)


class SfmReconstructor:
    """Runs pycolmap SIFT + sequential matching + incremental mapping on a keyframe folder.

    PINHOLE with one shared camera: a phone's main lens with stabilisation off is
    close enough to distortion-free that the splat trainer can then use plain K.
    """

    def run(self, image_dir: Path, work_dir: Path) -> SfmResult:
        """Reconstruct, keep the largest model, and load its registered images.

        Args:
            image_dir: keyframes from `KeyframeSampler`.
            work_dir: emptied and used for the COLMAP database and sparse model.

        Returns:
            The `SfmResult` for the largest reconstructed model.

        Raises:
            RuntimeError: if SfM registers no model at all.
        """
        import pycolmap

        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        database = work_dir / "database.db"
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "PINHOLE"
        pycolmap.extract_features(
            database, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader
        )
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 15
        # A turntable or orbit comes back to its start; loop closure needs these pairs too.
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(database, pairing_options=pairing)
        models = pycolmap.incremental_mapping(database, image_dir, work_dir / "sparse")
        if not models:
            raise RuntimeError("SfM registered no images; orbit too fast, too blurry or too textureless")
        model = largest_model(list(models.values()))
        print(f"[SfM] registered {model.num_reg_images()}/{len(list(image_dir.glob('*.png')))} images")
        return sfm_result_from_model(model, image_dir)


def largest_model(models: list):
    """The pycolmap Reconstruction that registered the most images (COLMAP may split an orbit)."""
    return max(models, key=lambda m: m.num_reg_images())


def load_sparse_models(sparse_dir: Path) -> list:
    """Every pycolmap Reconstruction under a COLMAP `sparse/` folder (one sub-folder per model).

    Args:
        sparse_dir: e.g. `data/scans/<name>/colmap/sparse`.
    """
    import pycolmap

    return [pycolmap.Reconstruction(str(p)) for p in sorted(sparse_dir.iterdir())]


def sfm_result_from_model(model, image_dir: Path | None = None) -> SfmResult:
    """Convert a pycolmap Reconstruction into numpy `PosedImage`s and points.

    Args:
        model: a registered reconstruction.
        image_dir: keyframe folder to read pixels from. If None, pixels are left as a
            1x1 placeholder: enough for anything that reads only poses and points
            (`object_frame_from_orbit`, plotting), not for splat training.
    """
    images = []
    for image in sorted(model.images.values(), key=lambda im: im.name):
        if not image.has_pose:
            continue
        t_cam_world = np.eye(4)
        t_cam_world[:3, :] = image.cam_from_world().matrix()
        if image_dir is None:
            rgb = np.zeros((1, 1, 3), dtype=np.float32)
        else:
            bgr = cv2.imread(str(image_dir / image.name))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(PosedImage(rgb, t_cam_world, np.asarray(image.camera.calibration_matrix())))
    points = np.array([p.xyz for p in model.points3D.values()])
    colors = np.array([p.color for p in model.points3D.values()], dtype=np.float32) / 255.0
    return SfmResult(images=images, points_world=points, colors=colors)


def object_frame_from_orbit(sfm: SfmResult) -> np.ndarray:
    """Similarity-free rigid frame for an orbit with no ChArUco: z up the orbit axis.

    The camera centres of an orbit (or a turntable, seen from the object) lie
    roughly on a circle; its plane normal is the object's vertical axis. The
    sign is chosen so cameras sit above the object. Origin: the sparse points'
    centroid in x/y, their 2nd-percentile height in z (the object's base).
    Scale stays SfM-arbitrary — `SplatFit.MATCH_BODY_XY` fixes it at attach
    time; the real capture uses a ChArUco board for metric scale instead.

    Args:
        sfm: the reconstruction.

    Returns:
        T_obj_world, 4x4, SfM frame -> object frame.
    """
    centres = sfm.camera_centres_world
    centred = centres - centres.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    z_obj = vt[2]
    points_centroid = np.median(sfm.points_world, axis=0)
    if np.dot(centres.mean(axis=0) - points_centroid, z_obj) < 0:
        z_obj = -z_obj
    x_obj = vt[0]
    y_obj = np.cross(z_obj, x_obj)
    rot_obj_world = np.stack([x_obj, y_obj, z_obj])

    points_obj = sfm.points_world @ rot_obj_world.T
    origin_obj = np.array([*np.median(points_obj[:, :2], axis=0), np.percentile(points_obj[:, 2], 2)])
    t_obj_world = np.eye(4)
    t_obj_world[:3, :3] = rot_obj_world
    t_obj_world[:3, 3] = -origin_obj
    return t_obj_world


@dataclass
class CropCylinder:
    """Vertical cylinder in a z-up frame, used to cut objects out of a splat.

    Attributes:
        radius: in the frame's units.
        z_min: bottom of the cylinder.
        z_max: top of the cylinder.
        centre_xy: shape (2,), axis position.
    """

    radius: float
    z_min: float
    z_max: float
    centre_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @classmethod
    def around_points(cls, points_obj: np.ndarray, margin: float = 0.15) -> CropCylinder:
        """Fit to the bulk of the SfM points (which sit on the textured object, not the backdrop).

        Args:
            points_obj: shape (P, 3) sparse points in the object frame (base at z=0, axis at x=y=0).
            margin: relative padding on radius and height.
        """
        radius = float(np.percentile(np.linalg.norm(points_obj[:, :2], axis=1), 95)) * (1 + margin)
        height = float(np.percentile(points_obj[:, 2], 98)) * (1 + margin)
        return cls(radius=radius, z_min=-margin * height, z_max=height)

    def contains(self, xyz: np.ndarray) -> np.ndarray:
        """Boolean mask of the points inside the cylinder."""
        return (
            (np.linalg.norm(xyz[:, :2] - self.centre_xy, axis=1) <= self.radius)
            & (xyz[:, 2] >= self.z_min)
            & (xyz[:, 2] <= self.z_max)
        )


def crop_ply(ply_in: Path, ply_out: Path, cylinders: list[CropCylinder], keep_inside: bool = True) -> int:
    """Filter a 3DGS .ply by cylinders: keep Gaussians inside any of them, or only those outside all.

    Args:
        ply_in: source splat, already in the cylinders' frame.
        ply_out: where to write the result (may equal `ply_in`).
        cylinders: the crop regions.
        keep_inside: True to cut objects out; False to remove them (e.g. from the table splat).

    Returns:
        Number of Gaussians kept.
    """
    from plyfile import PlyData, PlyElement

    vertices = PlyData.read(str(ply_in))["vertex"].data
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    inside = np.any([c.contains(xyz) for c in cylinders], axis=0)
    kept = vertices[inside if keep_inside else ~inside]
    PlyData([PlyElement.describe(kept, "vertex")]).write(str(ply_out))
    return len(kept)


def split_tabletop_background(
    table_ply: Path,
    background_ply: Path,
    radius_m: float,
    centre_xy: tuple[float, float] = (0.0, 0.0),
    z_min_m: float = -0.02,
    z_max_m: float = 0.6,
) -> tuple[int, int]:
    """Split a metric table-frame splat into tabletop (in place) and background (written aside).

    The tabletop is a vertical cylinder over the table; everything outside it (walls,
    floor, clutter beyond the table edge, and anything above `z_max_m`) is background.

    Args:
        table_ply: splat in the z-up table frame; rewritten to hold only the tabletop.
        background_ply: where the remaining Gaussians are written.
        radius_m: cylinder radius, in metres, around `centre_xy`.
        centre_xy: table-frame (x, y) of the table centre, in metres.
        z_min_m: cylinder bottom; just under the table surface, so legs and floor go to background.
        z_max_m: cylinder top, above the tallest object.

    Returns:
        (tabletop_count, background_count) Gaussians.
    """
    cylinder = CropCylinder(radius_m, z_min_m, z_max_m, np.array(centre_xy, dtype=float))
    background = crop_ply(table_ply, background_ply, [cylinder], keep_inside=False)
    tabletop = crop_ply(table_ply, table_ply, [cylinder], keep_inside=True)
    return tabletop, background


def border_colour(images_rgb: list[np.ndarray], border_px: int = 8) -> tuple[float, float, float]:
    """Median RGB of the image borders: the backdrop colour of a turntable/orbit clip."""
    strips = []
    for image in images_rgb:
        strips += [image[:border_px].reshape(-1, 3), image[-border_px:].reshape(-1, 3)]
        strips += [image[:, :border_px].reshape(-1, 3), image[:, -border_px:].reshape(-1, 3)]
    r, g, b = np.median(np.concatenate(strips), axis=0)
    return float(r), float(g), float(b)


@dataclass
class Similarity:
    """p_dst = scale * rotation @ p_src + translation.

    Attributes:
        scale: uniform scale factor (SfM units -> metres when aligning to a board).
        rotation: shape (3, 3).
        translation: shape (3,).
    """

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 3) points from the source to the destination frame."""
        return self.scale * points @ self.rotation.T + self.translation


def umeyama(src: np.ndarray, dst: np.ndarray) -> Similarity:
    """Least-squares similarity with dst ≈ s R src + t (Umeyama 1991).

    Args:
        src: shape (N, 3), N >= 3 non-collinear points.
        dst: shape (N, 3) corresponding points.

    Returns:
        The fitted `Similarity`.
    """
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    u, sigma, vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))  # guard against a reflection
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(sigma) @ d) / src_c.var(axis=0).sum())
    return Similarity(scale, rotation, mu_dst - scale * rotation @ mu_src)


def triangulate(projections: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation of one point from >= 2 views.

    Args:
        projections: 3x4 matrices K [R | t], world -> pixels.
        pixels: matching (u, v) observations.

    Returns:
        shape (3,) point in the world frame.
    """
    rows = []
    for p, (u, v) in zip(projections, pixels, strict=True):
        rows += [u * p[2] - p[0], v * p[2] - p[1]]
    _, _, vt = np.linalg.svd(np.array(rows))
    return vt[-1, :3] / vt[-1, 3]


class ChArUcoAligner:
    """Metric similarity from the SfM frame to a ChArUco board's frame (PolaRiS-style scale anchor).

    The board lies flat on the table in the orbit video. Its inner corners are
    detected in every posed keyframe, triangulated in the SfM frame with the
    SfM poses, and matched to their known metric board coordinates. Board frame:
    origin at a board corner, x/y in the board plane, z = board normal (OpenCV's
    convention; the board's own z points into the table, so callers flip it if
    they want z up — see `align`).
    """

    def __init__(self, board: cv2.aruco.CharucoBoard, min_views: int = 3) -> None:
        """Set the board definition, which must match the printed board exactly.

        Args:
            board: e.g. `ChArUcoCalibrator().board` from `camera.py`.
            min_views: views a corner must be seen in to be triangulated.
        """
        self._board = board
        self._detector = cv2.aruco.CharucoDetector(board)
        self._min_views = min_views

    def align(self, sfm: SfmResult) -> tuple[Similarity, float]:
        """Fit SfM frame -> table frame, where the table frame is the board frame flipped to z up.

        Args:
            sfm: posed keyframes in which the board is visible.

        Returns:
            (similarity, rms_residual_m): the transform, and the RMS distance in
            metres between triangulated and true corner positions after fitting.

        Raises:
            RuntimeError: if fewer than 6 corners could be triangulated.
        """
        observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for image in sfm.images:
            gray = cv2.cvtColor((image.image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            corners, ids, _, _ = self._detector.detectBoard(gray)
            if ids is None:
                continue
            projection = image.k @ image.t_cam_world[:3, :]
            for corner_id, pixel in zip(ids.ravel(), corners.reshape(-1, 2), strict=True):
                observations.setdefault(int(corner_id), []).append((projection, pixel))

        board_corners = self._board.getChessboardCorners()
        ids = [cid for cid, obs in observations.items() if len(obs) >= self._min_views]
        if len(ids) < 6:
            raise RuntimeError(f"only {len(ids)} ChArUco corners seen in >= {self._min_views} views; need 6")
        p_sfm = np.array([triangulate(*zip(*observations[cid], strict=True)) for cid in ids])
        # The board's own +z points into the table; flip y and z so table-frame z is up.
        p_table = board_corners[ids] * np.array([1.0, -1.0, -1.0])
        similarity = umeyama(p_sfm, p_table)
        rms = float(np.sqrt(np.mean(np.sum((similarity.apply(p_sfm) - p_table) ** 2, axis=1))))
        return similarity, rms


def transform_ply(ply_in: Path, ply_out: Path, similarity: Similarity) -> None:
    """Apply a similarity to a 3DGS .ply: means, log-scales and rotations.

    Args:
        ply_in: source splat.
        ply_out: destination (may equal `ply_in`).
        similarity: the transform to apply.
    """
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation

    vertices = PlyData.read(str(ply_in))["vertex"].data.copy()
    xyz = similarity.apply(np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1))
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    for axis in range(3):
        vertices[f"scale_{axis}"] += np.log(similarity.scale)  # stored as log-scale
    quats_wxyz = np.stack([vertices[f"rot_{i}"] for i in range(4)], axis=1)
    rotated = Rotation.from_matrix(similarity.rotation) * Rotation.from_quat(quats_wxyz, scalar_first=True)
    for i, column in enumerate(rotated.as_quat(scalar_first=True).T):
        vertices[f"rot_{i}"] = column
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(ply_out))


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Mean SSIM of two (1, 3, H, W) images in [0, 1], Gaussian window."""
    coords = torch.arange(window, device=img1.device, dtype=img1.dtype) - window // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = (g / g.sum())[None, :]
    kernel = (g.T @ g).expand(3, 1, window, window)

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=window // 2, groups=3)

    mu1, mu2 = blur(img1), blur(img2)
    sigma1 = blur(img1 * img1) - mu1**2
    sigma2 = blur(img2 * img2) - mu2**2
    sigma12 = blur(img1 * img2) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    return ssim_map.mean()


@dataclass
class TrainResult:
    """Outcome of one splat fit.

    Attributes:
        mean_psnr_db: mean PSNR over all training views after the last step.
        num_gaussians: Gaussians in the exported splat.
        previews: (render, photo) RGB float pairs for a few spread-out training views.
    """

    mean_psnr_db: float
    num_gaussians: int
    previews: list[tuple[np.ndarray, np.ndarray]]


class SplatTrainer:
    """Minimal 3DGS optimisation with gsplat: L1 + D-SSIM loss, gsplat's DefaultStrategy densification.

    Hyperparameters follow gsplat's `simple_trainer.py` defaults, scaled down
    in iterations: this fits a tabletop in minutes, not a benchmark scene.
    """

    def __init__(
        self,
        iterations: int = 7000,
        sh_degree: int = 0,
        ssim_weight: float = 0.2,
        background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: str = "cuda",
    ) -> None:
        """Configure training.

        Args:
            iterations: optimisation steps (one random image per step).
            sh_degree: spherical-harmonics degree; 0 is view-independent colour,
                which is all `SplatCompositor` uses.
            ssim_weight: weight of the D-SSIM term against L1.
            background_rgb: colour behind the Gaussians; white suits a turntable
                against a white backdrop, since it needs no Gaussians to explain.
            device: torch device.
        """
        self._iterations = iterations
        self._sh_degree = sh_degree
        self._ssim_weight = ssim_weight
        self._background = torch.tensor(background_rgb, device=device)[None]
        self._device = device

    def train(self, sfm: SfmResult, t_obj_world: np.ndarray, out_ply: Path) -> TrainResult:
        """Fit Gaussians to the posed images, in the object frame, and export a 3DGS .ply.

        Args:
            sfm: posed keyframes and sparse points (SfM frame).
            t_obj_world: SfM frame -> object frame; training happens directly in
                the object frame so the exported splat needs no re-posing.
            out_ply: where to write the trained splat.

        Returns:
            Fit quality and preview pairs (training views, so an optimistic check).
        """
        from gsplat import export_splats
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy

        dev = self._device
        t_world_obj = np.linalg.inv(t_obj_world)
        # Rigid change of frame: rotation part has unit scale, so means/scales stay consistent.
        points_obj = sfm.points_world @ t_obj_world[:3, :3].T + t_obj_world[:3, 3]
        viewmats = torch.tensor(
            np.stack([im.t_cam_world @ t_world_obj for im in sfm.images]), dtype=torch.float32, device=dev
        )
        ks = torch.tensor(np.stack([im.k for im in sfm.images]), dtype=torch.float32, device=dev)
        targets = [torch.from_numpy(im.image_rgb).to(dev) for im in sfm.images]
        height, width = sfm.images[0].image_rgb.shape[:2]

        camera_centres_obj = torch.linalg.inv(viewmats)[:, :3, 3]
        scene_scale = float((camera_centres_obj - camera_centres_obj.mean(0)).norm(dim=1).max()) * 1.1

        params, optimizers = self._init_params(points_obj, sfm.colors, scene_scale)
        strategy = DefaultStrategy(refine_stop_iter=int(self._iterations * 0.8), verbose=False)
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=scene_scale)
        means_lr_decay = 0.01 ** (1.0 / self._iterations)

        rng = np.random.default_rng(0)
        for step in range(self._iterations):
            i = int(rng.integers(len(targets)))
            render, info = self._render(
                params, viewmats[i : i + 1], ks[i : i + 1], width, height, rasterization
            )
            target = targets[i]
            l1 = (render - target).abs().mean()
            loss = (1 - self._ssim_weight) * l1 + self._ssim_weight * (
                1 - _ssim(render.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
            )
            strategy.step_pre_backward(params, optimizers, state, step, info)
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for group in optimizers["means"].param_groups:
                group["lr"] *= means_lr_decay
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
            if step % 1000 == 0 or step == self._iterations - 1:
                print(f"[splat] step {step:5d} loss {loss.item():.4f} gaussians {len(params['means'])}")

        with torch.no_grad():
            psnrs, previews = [], []
            preview_ids = set(np.linspace(0, len(targets) - 1, 4).astype(int).tolist())
            for i, target in enumerate(targets):
                render, _ = self._render(
                    params, viewmats[i : i + 1], ks[i : i + 1], width, height, rasterization
                )
                mse = float(((render - target) ** 2).mean())
                psnrs.append(-10 * math.log10(max(mse, 1e-10)))
                if i in preview_ids:
                    previews.append((render.cpu().numpy(), target.cpu().numpy()))
            out_ply.parent.mkdir(parents=True, exist_ok=True)
            export_splats(
                means=params["means"],
                scales=params["scales"],
                quats=F.normalize(params["quats"], dim=-1),
                opacities=params["opacities"],
                sh0=params["sh0"],
                shN=params["shN"],
                format="ply",
                save_to=str(out_ply),
            )
        return TrainResult(float(np.mean(psnrs)), len(params["means"]), previews)

    def _init_params(
        self, points: np.ndarray, colors: np.ndarray, scene_scale: float
    ) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer]]:
        """Gaussians at the SfM points: isotropic scale from 3-NN distance, SH DC from point colour."""
        dev = self._device
        means = torch.tensor(points, dtype=torch.float32, device=dev)
        dists = torch.cdist(means, means)
        knn = dists.topk(4, largest=False).values[:, 1:]
        log_scale = torch.log(knn.mean(dim=1).clamp_min(1e-7))[:, None].repeat(1, 3)
        n = len(means)
        sh_c0 = 0.28209479177387814  # zeroth SH basis constant: colour = sh0 * C0 + 0.5
        sh0 = ((torch.tensor(colors, device=dev) - 0.5) / sh_c0)[:, None, :]
        shn = torch.zeros(n, (self._sh_degree + 1) ** 2 - 1, 3, device=dev)
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(log_scale),
                "quats": torch.nn.Parameter(torch.rand(n, 4, device=dev)),
                "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=dev))),
                "sh0": torch.nn.Parameter(sh0),
                "shN": torch.nn.Parameter(shn),
            }
        )
        learning_rates = {
            "means": 1.6e-4 * scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }
        optimizers = {
            name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15)
            for name, lr in learning_rates.items()
        }
        return params, optimizers

    def _render(self, params, viewmat, k, width, height, rasterization) -> tuple[torch.Tensor, dict]:
        """Render one view: (H, W, 3) RGB over the background colour, plus gsplat's info dict."""
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        renders, _alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=k,
            width=width,
            height=height,
            sh_degree=self._sh_degree,
            backgrounds=self._background,
            packed=False,
        )
        return renders[0].clamp(0, 1), info
