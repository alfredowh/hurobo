"""Object crop for a single-object scan: RANSAC floor plane, then the cluster of points standing on it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from hurobo.scan import CropCylinder, SfmResult, Similarity, transform_ply


@dataclass
class FloorPlane:
    """A fitted floor in a z-up frame.

    Attributes:
        normal: shape (3,), unit, pointing up (positive z component).
        point: shape (3,), a point on the plane.
        inlier_fraction: share of the fitted points within the inlier distance.
        noise_sigma: robust standard deviation of the inliers' distance to the plane.
    """

    normal: np.ndarray
    point: np.ndarray
    inlier_fraction: float
    noise_sigma: float

    def heights(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of (P, 3) points above the plane (negative below)."""
        return (points - self.point) @ self.normal

    def z_at(self, xy: np.ndarray) -> float:
        """Height of the plane (frame z) above the horizontal position `xy`."""
        nx, ny, nz = self.normal
        return float(self.point[2] - (nx * (xy[0] - self.point[0]) + ny * (xy[1] - self.point[1])) / nz)

    @property
    def tilt_deg(self) -> float:
        """Angle between the plane's normal and the frame's +z axis."""
        return float(np.degrees(np.arccos(np.clip(self.normal[2], -1.0, 1.0))))


@dataclass
class ObjectCrop:
    """A crop cylinder estimated from the scene geometry, with what it was derived from.

    Attributes:
        cylinder: the crop, in the same z-up frame as the input points.
        floor: the fitted floor.
        ring_centre_xy: centre of the circle the cameras move on.
        ring_radius: radius of that circle.
        num_candidates: points above the floor and inside the camera ring.
        num_cluster_points: points in the cluster the cylinder was fitted to.
    """

    cylinder: CropCylinder
    floor: FloorPlane
    ring_centre_xy: np.ndarray
    ring_radius: float
    num_candidates: int
    num_cluster_points: int


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle through (N, 2) points (Kåsa fit): returns (centre, radius).

    Works for a partial arc as well as a full loop.
    """
    design = np.c_[2 * xy, np.ones(len(xy))]
    target = (xy**2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(design, target, rcond=None)[0]
    return np.array([cx, cy]), float(np.sqrt(c + cx**2 + cy**2))


def fit_floor_plane(
    points: np.ndarray,
    inlier_distance: float,
    iterations: int = 1000,
    min_inlier_fraction: float = 0.3,
    seed: int = 0,
) -> FloorPlane | None:
    """Dominant plane of a point cloud by RANSAC, refined by least squares on its inliers.

    Each trial takes three random points, builds their plane and counts the points within
    `inlier_distance`. The floor is the largest coplanar set, so the trial that lands three
    points on it wins; the object and far-away clutter are outliers and do not pull the fit.

    Args:
        points: shape (P, 3), in a z-up frame.
        inlier_distance: plane-to-point distance that counts as "on the plane", in the points'
            units. Tie it to the scene (e.g. the camera orbit radius), not to the cloud's
            extent, which stray far points inflate.
        iterations: random triples tried.
        min_inlier_fraction: below this share of inliers there is no dominant plane.
        seed: RNG seed, for reproducible results.

    Returns:
        The plane, or None when no plane holds `min_inlier_fraction` of the points.
    """
    rng = np.random.default_rng(seed)
    # A fixed point order makes the seeded triples, and so the result, independent of how the
    # points were enumerated (COLMAP does not return them in a stable order).
    points = points[np.lexsort(points.T)]
    best_count, best = 0, None
    for _ in range(iterations):
        triple = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(triple[1] - triple[0], triple[2] - triple[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        count = int((np.abs((points - triple[0]) @ normal) < inlier_distance).sum())
        if count > best_count:
            best_count, best = count, (normal, triple[0])
    if best is None or best_count / len(points) < min_inlier_fraction:
        return None
    normal, origin = best
    inliers = np.abs((points - origin) @ normal) < inlier_distance
    centre = points[inliers].mean(axis=0)
    normal = np.linalg.svd(points[inliers] - centre)[2][2]
    if normal[2] < 0:
        normal = -normal
    distances = (points[inliers] - centre) @ normal
    sigma = 1.4826 * float(np.median(np.abs(distances - np.median(distances))))
    return FloorPlane(normal, centre, float(inliers.mean()), sigma)


def largest_cluster_mask(points: np.ndarray, link_factor: float) -> np.ndarray:
    """Boolean mask of the largest group of points that are chained together by short links.

    Two points are linked when closer than `link_factor` times the median nearest-neighbour spacing.

    Args:
        points: shape (P, 3).
        link_factor: link distance in units of the median nearest-neighbour spacing.
    """
    tree = cKDTree(points)
    spacing = float(np.median(tree.query(points, k=2)[0][:, 1]))
    pairs = tree.query_pairs(link_factor * spacing, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels == np.bincount(labels).argmax()


def estimate_object_crop(
    points: np.ndarray,
    camera_centres: np.ndarray,
    margin: float = 1.3,
    ring_fraction: float = 0.6,
    floor_distance_fraction: float = 0.01,
    height_sigmas: float = 5.0,
    link_factor: float = 5.0,
    min_points: int = 30,
    seed: int = 0,
) -> ObjectCrop | None:
    """Crop cylinder around the object standing on the floor in the middle of the camera orbit.

    1. The camera orbit circle gives the ring the object sits inside.
    2. RANSAC fits the floor plane to the points.
    3. Points clearly above the floor and inside the ring are the object candidates.
    4. The largest connected cluster of candidates is the object; the cylinder is fitted to it.

    Assumes a dominant floor or table plane and one object near the orbit centre; where that
    does not hold it returns None and the caller falls back to another method.

    Args:
        points: shape (P, 3) sparse SfM points in a z-up frame (e.g. the object frame).
        camera_centres: shape (N, 3) registered camera centres in the same frame.
        margin: multiplies the cluster's radius, so the crop keeps a rim around the object.
        ring_fraction: candidates must lie within this fraction of the ring radius of its centre.
        floor_distance_fraction: floor inlier distance as a fraction of the ring radius.
        height_sigmas: candidates must be this many floor-noise sigmas above the floor.
        link_factor: points closer than this many median neighbour spacings are one cluster.
        min_points: fewer candidates than this is "no object found".
        seed: RNG seed for RANSAC.

    Returns:
        The estimated crop, or None if there is no dominant floor or no object cluster.
    """
    ring_centre, ring_radius = fit_circle(camera_centres[:, :2])
    # Fit only inside the camera ring: a scan of a room (cabinets, walls) has most of its points off the
    # floor, and the plate clip's 15 % floor share fell under the 30 % bar until the fit was restricted.
    within_ring = np.linalg.norm(points[:, :2] - ring_centre, axis=1) < ring_radius
    floor = fit_floor_plane(points[within_ring], floor_distance_fraction * ring_radius, seed=seed)
    if floor is None:
        return None
    min_height = max(height_sigmas * floor.noise_sigma, 0.02 * ring_radius)
    heights = floor.heights(points)
    inside_ring = np.linalg.norm(points[:, :2] - ring_centre, axis=1) < ring_fraction * ring_radius
    candidates = inside_ring & (heights > min_height)
    if candidates.sum() < min_points:
        return None

    cloud = points[candidates]
    in_cluster = largest_cluster_mask(cloud, link_factor)
    if in_cluster.sum() < min_points:
        return None

    cluster = cloud[in_cluster]
    centre_xy = np.median(cluster[:, :2], axis=0)
    radius = margin * float(np.percentile(np.linalg.norm(cluster[:, :2] - centre_xy, axis=1), 90))
    height = float(np.percentile(floor.heights(cluster), 95))
    floor_z = floor.z_at(centre_xy)
    cylinder = CropCylinder(radius, floor_z - 0.25 * height, floor_z + 1.3 * height, centre_xy)
    return ObjectCrop(cylinder, floor, ring_centre, ring_radius, int(candidates.sum()), int(in_cluster.sum()))


@dataclass
class ObjectSplat:
    """What `extract_object_splat` kept and how it is framed.

    Attributes:
        num_gaussians: Gaussians in the output.
        cut_height: Gaussians at or below this height above the floor were dropped (SfM units).
        footprint_radius: 95th-percentile horizontal distance of the kept Gaussians from the axis.
        height: 98th-percentile height of the kept Gaussians above z = 0.
        tilt_removed_deg: how far the floor normal was rotated onto +z.
    """

    num_gaussians: int
    cut_height: float
    footprint_radius: float
    height: float
    tilt_removed_deg: float


def extract_object_splat(
    ply_in: Path,
    ply_out: Path,
    floor: FloorPlane,
    cylinder: CropCylinder,
    height_sigmas: float = 3.0,
    min_opacity: float = 0.05,
    link_factor: float = 8.0,
) -> ObjectSplat:
    """The object alone, ready to attach to a simulator body: floor removed, upright, base at z = 0.

    1. Keep Gaussians inside `cylinder` (horizontally) and opaque enough.
    2. Drop everything within `height_sigmas` of the floor level; that scatter is measured on the
       Gaussians themselves (they spread more than the SfM points).
    3. Keep the largest connected blob, which discards floaters.
    4. Rotate the floor normal onto +z, put the floor under the blob's centre at the origin.

    The floor shadow is baked into the floor Gaussians, so it goes with them. Scale stays in SfM
    units, so attach with `SplatFit.MATCH_BODY_XY`.

    Args:
        ply_in: uncropped 3DGS splat in the frame `floor` and `cylinder` are expressed in.
        ply_out: where to write the object splat.
        floor: the fitted floor in that frame.
        cylinder: the object crop; only its horizontal extent is used.
        height_sigmas: floor-level scatter multiples to cut at.
        min_opacity: Gaussians below this opacity are dropped. Keep it low: a smooth white object is
            drawn by many translucent Gaussians (median opacity 0.05 on the bowl), and 0.2 left a hollow rim.
        link_factor: blob linking distance in median nearest-neighbour spacings; wide enough to join
            the translucent surface patches of one object.

    Returns:
        A summary of what was kept.

    Raises:
        ValueError: if nothing stands above the floor inside the cylinder.
    """
    vertices = PlyData.read(str(ply_in))["vertex"].data.copy()
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"]))
    height = floor.heights(xyz)
    in_cylinder = (np.linalg.norm(xyz[:, :2] - cylinder.centre_xy, axis=1) <= cylinder.radius) & (
        opacity > min_opacity
    )
    # Floor-level scatter of the Gaussians. Start from the SfM points' scatter and re-measure inside a
    # +-4 sigma band until it settles, so it is the floor's own scatter and not the object's or floaters'.
    candidates = height[in_cylinder]
    level = 0.0  # the fitted plane itself
    sigma = max(floor.noise_sigma, 1e-6)
    for _ in range(8):
        band = candidates[np.abs(candidates - level) < 4.0 * sigma]
        level = float(np.median(band))
        sigma = 1.4826 * float(np.median(np.abs(band - level)))
    cut = level + height_sigmas * sigma
    standing = np.flatnonzero(in_cylinder & (height > cut))
    if len(standing) < 20:
        raise ValueError("no Gaussians stand above the floor inside the crop")
    keep = standing[largest_cluster_mask(xyz[standing], link_factor)]

    centre_xy = np.median(xyz[keep, :2], axis=0)
    anchor = np.array([centre_xy[0], centre_xy[1], floor.z_at(centre_xy)])
    rotation = Rotation.align_vectors([[0.0, 0.0, 1.0]], [floor.normal])[0].as_matrix()
    similarity = Similarity(1.0, rotation, -rotation @ anchor)

    subset = vertices[keep]
    scratch = ply_out.with_suffix(".tmp.ply")
    PlyData([PlyElement.describe(subset, "vertex")]).write(str(scratch))
    transform_ply(scratch, ply_out, similarity)
    scratch.unlink()

    moved = similarity.apply(xyz[keep])
    return ObjectSplat(
        num_gaussians=len(keep),
        cut_height=cut,
        footprint_radius=float(np.percentile(np.linalg.norm(moved[:, :2], axis=1), 95)),
        height=float(np.percentile(moved[:, 2], 98)),
        tilt_removed_deg=floor.tilt_deg,
    )


@dataclass
class CropChoice:
    """The crop selected for a scan and how it was chosen.

    Attributes:
        cylinder: the crop, in the object frame.
        method: "explicit", "floor + cluster", or "percentile of all points".
        floor: the fitted floor when one was found, even for an explicit crop; None otherwise.
    """

    cylinder: CropCylinder
    method: str
    floor: FloorPlane | None


def choose_object_crop(
    points: np.ndarray, camera_centres: np.ndarray, explicit: CropCylinder | None = None
) -> CropChoice:
    """The crop to use: explicit if given, else floor + cluster, else the old percentile fit.

    The floor is reported whenever one was found, so a splat can be cut off it even when the
    user supplied the cylinder.

    Args:
        points: shape (P, 3) sparse points in the object frame.
        camera_centres: shape (N, 3) camera centres in the object frame.
        explicit: a user-chosen cylinder; it always wins.
    """
    estimate = estimate_object_crop(points, camera_centres)
    floor = estimate.floor if estimate is not None else None
    if explicit is not None:
        return CropChoice(explicit, "explicit", floor)
    if estimate is not None:
        return CropChoice(estimate.cylinder, "floor + cluster", floor)
    # No dominant floor or no object cluster (e.g. a turntable on a plain backdrop): the original fit,
    # which assumes the sparse points sit on the object.
    return CropChoice(CropCylinder.around_points(points), "percentile of all points", None)


@dataclass
class WhiteObjectReference:
    """Where a white object is, triangulated from its white pixels in the keyframes.

    This is a validation reference, independent of the splat and the SfM points. It only works
    because the bowl is the one white object in view; the pipeline does not use it.

    Attributes:
        centre: shape (3,), in the frame of the `SfmResult` it was computed in.
        radius: object radius in that frame's units (half its apparent rim width).
        num_views: views that contributed.
        mean_ray_distance: mean distance of the centre to the viewing rays, same units.
    """

    centre: np.ndarray
    radius: float
    num_views: int
    mean_ray_distance: float


def locate_white_object(
    sfm: SfmResult, min_value: float = 0.7, max_saturation: float = 0.25
) -> WhiteObjectReference | None:
    """Triangulate the largest white blob of every keyframe into one 3D centre and a radius.

    Args:
        sfm: reconstruction with pixels (`sfm_result_from_model(model, image_dir)`).
        min_value: HSV value (brightness) a pixel needs to count as white, in [0, 1].
        max_saturation: HSV saturation a pixel may have to count as white, in [0, 1].

    Returns:
        The reference, or None if fewer than three views show a usable blob.
    """
    origins, directions, widths_px, focals = [], [], [], []
    for image in sfm.images:
        rgb = (np.clip(image.image_rgb, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] < max_saturation * 255) & (hsv[..., 2] > min_value * 255)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            continue
        blob = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[blob]
        height, width = mask.shape
        if area < 0.002 * height * width or x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue  # nothing there, or the blob is cut off by the image border
        u, v = centroids[blob]
        pixel_ray = np.linalg.inv(image.k) @ np.array([u, v, 1.0])
        t_world_cam = np.linalg.inv(image.t_cam_world)
        direction = t_world_cam[:3, :3] @ pixel_ray
        origins.append(t_world_cam[:3, 3])
        directions.append(direction / np.linalg.norm(direction))
        widths_px.append(float(w))
        focals.append(float(image.k[0, 0]))
    if len(origins) < 3:
        return None

    normal_matrix, rhs = np.zeros((3, 3)), np.zeros(3)
    for origin, direction in zip(origins, directions, strict=True):
        projector = np.eye(3) - np.outer(direction, direction)
        normal_matrix += projector
        rhs += projector @ origin
    centre = np.linalg.solve(normal_matrix, rhs)
    ray_distances = [
        float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (centre - o)))
        for o, d in zip(origins, directions, strict=True)
    ]
    radii = [
        0.5 * width * float(np.linalg.norm(centre - origin)) / focal
        for width, origin, focal in zip(widths_px, origins, focals, strict=True)
    ]
    return WhiteObjectReference(centre, float(np.median(radii)), len(origins), float(np.mean(ray_distances)))


def crop_report(
    methods: dict[str, CropCylinder], reference: WhiteObjectReference, ply_path: Path
) -> pd.DataFrame:
    """One row per crop method, judged against an independent reference for the object.

    Args:
        methods: label -> cylinder, all in the splat's (object) frame.
        reference: where the object really is, from `locate_white_object` in the same frame.
        ply_path: the uncropped splat, to count how many Gaussians each crop keeps.

    Returns:
        Radius relative to the object, how far the centre is off (in object radii), whether the
        object's footprint lies inside, and the share of the splat's Gaussians kept.
    """
    vertices = PlyData.read(str(ply_path))["vertex"].data
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    rows = []
    for label, cylinder in methods.items():
        offset = float(np.linalg.norm(reference.centre[:2] - cylinder.centre_xy))
        rows.append(
            {
                "method": label,
                "radius / object radius": round(cylinder.radius / reference.radius, 1),
                "centre off (object radii)": round(offset / reference.radius, 2),
                "object inside": bool(offset + reference.radius <= cylinder.radius),
                "Gaussians kept (%)": round(100 * float(cylinder.contains(xyz).mean()), 1),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ExpectedObject:
    """An object the table scan should contain, from a ruler measurement.

    Attributes:
        name: output stem, e.g. "plate".
        diameter_m: footprint diameter.
        height_m: height above the tabletop.
    """

    name: str
    diameter_m: float
    height_m: float


def locate_objects_on_table(
    table_ply: Path,
    expected: list[ExpectedObject],
    cell_m: float = 0.02,
    min_height_m: float = 0.012,
    max_height_m: float = 0.15,
    min_points_per_cell: int = 20,
    margin_m: float = 0.01,
    region_xy: tuple[float, float, float, float] | None = None,
) -> dict[str, CropCylinder]:
    """Find each expected object as a blob of Gaussians standing above the tabletop of a metric splat.

    Gaussians between `min_height_m` and `max_height_m` above z = 0 are binned into a top-down grid;
    connected dense cells form blobs. Each expected object takes the unused blob whose footprint
    diameter is closest to its measured one (largest objects first), so the plate and the bowl are
    told apart by size, not by position.

    Args:
        table_ply: splat in the z-up table frame with the tabletop at z = 0.
        expected: objects to find, with measured diameter and height.
        cell_m: grid cell size.
        min_height_m: ignore the tabletop's own scatter below this height.
        max_height_m: ignore floaters above this height.
        min_points_per_cell: occupancy threshold of a cell.
        margin_m: padding added to the measured radius and height of the returned cylinders.
        region_xy: (x_min, x_max, y_min, y_max) to search in, e.g. the tabletop box; without it
            floaters and clutter outside the table are candidates too.

    Returns:
        name -> cylinder (radius/height from the measurement, centre from the blob).

    Raises:
        RuntimeError: if fewer blobs than expected objects were found.
    """
    from scipy import ndimage

    vertices = PlyData.read(str(table_ply))["vertex"].data
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    keep = (xyz[:, 2] > min_height_m) & (xyz[:, 2] < max_height_m)
    if region_xy is not None:
        x_min, x_max, y_min, y_max = region_xy
        keep &= (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) & (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    above = xyz[keep]
    x_edges = np.arange(above[:, 0].min(), above[:, 0].max() + cell_m, cell_m)
    y_edges = np.arange(above[:, 1].min(), above[:, 1].max() + cell_m, cell_m)
    counts, _, _ = np.histogram2d(above[:, 0], above[:, 1], bins=[x_edges, y_edges])
    labels, num_blobs = ndimage.label(counts >= min_points_per_cell)

    blobs: list[tuple[np.ndarray, float]] = []  # (centre_xy, equivalent diameter)
    for blob_id in range(1, num_blobs + 1):
        cells_x, cells_y = np.nonzero(labels == blob_id)
        area = len(cells_x) * cell_m**2
        if area < 0.5 * np.pi * (0.03) ** 2:  # smaller than a 6 cm disc: scatter, not an object
            continue
        centre = np.array(
            [
                x_edges[cells_x].mean() + cell_m / 2,
                y_edges[cells_y].mean() + cell_m / 2,
            ]
        )
        blobs.append((centre, 2 * float(np.sqrt(area / np.pi))))
    if len(blobs) < len(expected):
        raise RuntimeError(f"found {len(blobs)} blobs above the table, expected {len(expected)} objects")

    cylinders: dict[str, CropCylinder] = {}
    for item in sorted(expected, key=lambda e: -e.diameter_m):
        best = min(range(len(blobs)), key=lambda i: abs(blobs[i][1] - item.diameter_m))
        centre, _ = blobs.pop(best)
        cylinders[item.name] = CropCylinder(
            item.diameter_m / 2 + margin_m, -0.01, item.height_m + margin_m, centre
        )
    return cylinders
