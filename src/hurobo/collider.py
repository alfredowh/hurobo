"""Splat -> surface mesh -> convex collision pieces -> patched LIBERO object asset.

The splat is only a picture; physics needs a solid. Recipe (PolaRiS's split of
"splat for rendering, mesh for collision"): render depth of the fitted splat from
many viewpoints, fuse it into a TSDF with Open3D, extract a triangle mesh, split
it into convex pieces with CoACD (MuJoCo only collides convex shapes), and write
those pieces over the original object's collision geoms in a *copy* of LIBERO's
asset folder. LIBERO itself is not modified: it is pointed at the copy.
"""

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh

from hurobo.splat import splat_centre_and_radius
from hurobo.utils.transforms import look_at_t_cam_world


@dataclass
class GaussiansBodyFrame:
    """A splat already fitted into its sim body's frame (output of `fit_gaussians_to_body`).

    Attributes:
        xyz: (N, 3) means.
        rot_wxyz: (N, 4) unit quaternions.
        scale: (N, 3) linear (not log) scales.
        opacity: (N,) in [0, 1].
        sh_dc: (N, 3) zeroth-order SH colour coefficients.
    """

    xyz: np.ndarray
    rot_wxyz: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh_dc: np.ndarray


class SplatSurfaceExtractor:
    """Depth-fusion mesh extraction from a splat (the 2DGS paper's bounded-mode method, on gsplat)."""

    def __init__(
        self,
        voxel_m: float = 0.002,
        trunc_m: float = 0.005,
        image_size: int = 320,
        num_azimuths: int = 16,
        elevations_deg: tuple[float, ...] = (8, 25, 45, 65, 85),
        device: str = "cuda",
    ) -> None:
        """Configure the fusion.

        Args:
            voxel_m: TSDF voxel size. Must be well below the thinnest wall to keep.
            trunc_m: TSDF truncation. Keep it under the wall thickness, or the
                inner and outer surfaces of a thin wall fuse into one slab.
            image_size: square depth image size in pixels.
            num_azimuths: viewpoints per elevation ring.
            elevations_deg: ring elevations above the object's equator plane.
            device: torch device for rasterization.
        """
        self._voxel_m = voxel_m
        self._trunc_m = trunc_m
        self._image_size = image_size
        self._num_azimuths = num_azimuths
        self._elevations_deg = elevations_deg
        self._device = device

    def extract(self, gaussians: GaussiansBodyFrame, min_alpha: float = 0.5) -> trimesh.Trimesh:
        """Fuse depth renders of the splat into a mesh, keeping the largest connected piece.

        Args:
            gaussians: the fitted splat.
            min_alpha: pixels with less accumulated opacity than this have no depth.

        Returns:
            The surface mesh in the splat's (body) frame.
        """
        import open3d as o3d
        from gsplat.rendering import rasterization

        centre, radius = splat_centre_and_radius(gaussians.xyz, gaussians.opacity)
        fov_deg = 40.0
        focal = (self._image_size / 2) / np.tan(np.radians(fov_deg / 2))
        distance = 1.3 * radius / np.tan(np.radians(fov_deg / 2))

        dev = self._device
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)  # noqa: E731
        means, quats, scales = to_t(gaussians.xyz), to_t(gaussians.rot_wxyz), to_t(gaussians.scale)
        opacities, colors = to_t(gaussians.opacity), to_t(gaussians.sh_dc)[:, None, :]
        k = np.array([[focal, 0, self._image_size / 2], [0, focal, self._image_size / 2], [0, 0, 1.0]])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self._image_size, self._image_size, focal, focal, self._image_size / 2, self._image_size / 2
        )
        # Dense volume, not ScalableTSDFVolume: in the installed open3d 0.20 wheel the sparse one
        # returned zero triangles even for a flat synthetic plane (checked), the dense one is fine.
        length = 2.2 * radius
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=length,
            resolution=int(np.ceil(length / self._voxel_m)),
            sdf_trunc=self._trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=centre - length / 2,
        )

        for elevation in np.radians(self._elevations_deg):
            for azimuth in np.linspace(0, 2 * np.pi, self._num_azimuths, endpoint=False):
                eye = centre + distance * np.array(
                    [
                        np.cos(elevation) * np.cos(azimuth),
                        np.cos(elevation) * np.sin(azimuth),
                        np.sin(elevation),
                    ]
                )
                t_cam_world = look_at_t_cam_world(eye, centre)
                with torch.no_grad():
                    render, alpha, _ = rasterization(
                        means=means,
                        quats=quats,
                        scales=scales,
                        opacities=opacities,
                        colors=colors,
                        viewmats=to_t(t_cam_world)[None],
                        Ks=to_t(k)[None],
                        width=self._image_size,
                        height=self._image_size,
                        sh_degree=0,
                        render_mode="RGB+ED",
                    )
                rgb = (render[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                depth = render[0, ..., 3].cpu().numpy().astype(np.float32)
                depth[alpha[0, ..., 0].cpu().numpy() < min_alpha] = 0.0
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb),
                    o3d.geometry.Image(depth),
                    depth_scale=1.0,
                    depth_trunc=2.5 * distance,
                    convert_rgb_to_intensity=False,
                )
                volume.integrate(rgbd, intrinsic, t_cam_world)

        mesh_o3d = volume.extract_triangle_mesh()
        mesh = trimesh.Trimesh(np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles), process=True)
        if len(mesh.faces) == 0:
            raise RuntimeError("depth fusion produced an empty mesh; is the splat opaque and non-empty?")
        pieces = mesh.split(only_watertight=False)
        return max(pieces, key=lambda m: len(m.faces))


def decompose_convex(
    mesh: trimesh.Trimesh,
    max_pieces: int = 24,
    threshold: float = 0.04,
    preprocess_resolution: int = 120,
) -> list[trimesh.Trimesh]:
    """Split a (possibly concave, non-watertight) mesh into convex pieces with CoACD.

    Args:
        mesh: the surface mesh.
        max_pieces: cap on the number of convex pieces (LIBERO's own bowl uses ~40 boxes).
        threshold: CoACD concavity threshold; lower = tighter fit, more pieces.
        preprocess_resolution: voxel resolution of CoACD's manifold repair. A thin
            wall needs enough resolution to survive it.

    Returns:
        Convex trimesh pieces, in the mesh's frame.
    """
    import coacd

    result = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold,
        max_convex_hull=max_pieces,
        preprocess_mode="on",
        preprocess_resolution=preprocess_resolution,
        seed=0,
    )
    return [trimesh.Trimesh(np.asarray(v), np.asarray(f)) for v, f in result]


def rim_grasp_offset(mesh: trimesh.Trimesh, depth_below_rim_m: float = 0.019) -> np.ndarray:
    """Where to pinch the bowl wall on its +y side, in the bowl's frame (x=0 is the bowl centre).

    The top ring of the surface mesh is the rim. Taking the median y of the mesh
    points near the top at x≈0 gives the wall's mid-line (the mesh has an inner
    and an outer surface). `depth_below_rim_m` copies the offset the stock-bowl
    demo used (19 mm under the rim), so a fingertip sits on the wall, not above it.

    Args:
        mesh: the surface mesh in the body frame, bowl axis = z.
        depth_below_rim_m: how far below the rim to place the fingertips.

    Returns:
        shape (3,) grasp point in the body frame.
    """
    z_top = float(np.percentile(mesh.vertices[:, 2], 99.5))
    band = mesh.vertices[
        (mesh.vertices[:, 2] > z_top - 0.02)
        & (np.abs(mesh.vertices[:, 0]) < 0.015)
        & (mesh.vertices[:, 1] > 0)
    ]
    if len(band) == 0:
        raise RuntimeError("no rim points found on the +y side; is the mesh upright and centred?")
    return np.array([0.0, float(np.median(band[:, 1])), z_top - depth_below_rim_m])


@dataclass
class ColliderPatchResult:
    """What `patch_object_xml` did.

    Attributes:
        num_removed_geoms: original collision geoms replaced.
        num_pieces: convex pieces written.
        collision_mass_kg: mass distributed over the new pieces (equals the old collision geoms' mass).
    """

    num_removed_geoms: int
    num_pieces: int
    collision_mass_kg: float


def patch_object_xml(
    xml_in: Path, xml_out: Path, pieces: list[trimesh.Trimesh], collision_dir_name: str = "collision"
) -> ColliderPatchResult:
    """Replace a LIBERO object's group-0 collision geoms with convex mesh pieces.

    Everything else in the file (visual mesh, texture, sites, body names) is
    kept, so LIBERO's init states, predicates and our joint-name lookups still work.
    Contact parameters (friction, solref, solimp) are copied from the geoms being
    replaced. Mass is conserved: the old boxes' total mass is split over the new
    pieces by hull volume, since a hull of a thin wall chunk is far bulkier than the wall.

    Args:
        xml_in: the object's original MJCF, e.g. `.../akita_black_bowl/akita_black_bowl.xml`.
        xml_out: where to write the patched MJCF; pieces go to a sibling `collision/` folder.
        pieces: convex meshes in the object's body frame, metres.
        collision_dir_name: sub-folder name for the piece `.obj` files.

    Returns:
        A `ColliderPatchResult`.
    """
    tree = ET.parse(xml_in)
    root = tree.getroot()
    asset = root.find("asset")
    object_body = root.find(".//body[@name='object']")
    if asset is None or object_body is None:
        raise ValueError(f"{xml_in} has no <asset> or <body name='object'>")

    old = [g for g in object_body.findall("geom") if g.get("group") == "0"]
    if not old:
        raise ValueError(f"{xml_in} has no group-0 collision geoms to replace")
    contact_attrs = {k: v for k, v in old[0].attrib.items() if k in ("friction", "solref", "solimp")}
    density = float(old[0].get("density", "1000"))
    old_mass = sum(
        8 * np.prod([float(v) for v in g.get("size").split()]) for g in old if g.get("type") == "box"
    )
    old_mass *= density
    for geom in old:
        object_body.remove(geom)

    piece_dir = xml_out.parent / collision_dir_name
    piece_dir.mkdir(parents=True, exist_ok=True)
    volumes = np.array([max(float(p.convex_hull.volume), 1e-12) for p in pieces])
    for i, (piece, volume) in enumerate(zip(pieces, volumes, strict=True)):
        mesh_name = f"{xml_out.stem}_col_{i:02d}"
        piece.export(piece_dir / f"{mesh_name}.obj")
        ET.SubElement(asset, "mesh", file=f"{collision_dir_name}/{mesh_name}.obj", name=mesh_name)
        ET.SubElement(
            object_body,
            "geom",
            type="mesh",
            mesh=mesh_name,
            group="0",
            rgba="0.8 0.8 0.8 0.3",
            mass=f"{old_mass * volume / volumes.sum():.8f}",
            **contact_attrs,
        )
    ET.indent(tree)
    tree.write(xml_out)
    return ColliderPatchResult(len(old), len(pieces), float(old_mass))


def build_asset_overlay(
    original_assets: Path, overlay_dir: Path, object_folder: str, patched_object_dir: Path
) -> Path:
    """A copy of LIBERO's asset tree that differs only in one scanned object, built from symlinks.

    Args:
        original_assets: LIBERO's real assets directory.
        overlay_dir: where to build the overlay (emptied first).
        object_folder: name under `stable_scanned_objects/`, e.g. "akita_black_bowl".
        patched_object_dir: the folder holding the patched XML and its files; used
            in place of the original object folder.

    Returns:
        `overlay_dir`, ready for `use_assets_dir`.
    """
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir(parents=True)
    scanned = "stable_scanned_objects"
    for entry in original_assets.iterdir():
        if entry.name != scanned:
            os.symlink(entry.resolve(), overlay_dir / entry.name)
    (overlay_dir / scanned).mkdir()
    for entry in (original_assets / scanned).iterdir():
        if entry.name != object_folder:
            os.symlink(entry.resolve(), overlay_dir / scanned / entry.name)
    shutil.copytree(patched_object_dir, overlay_dir / scanned / object_folder, symlinks=True)
    return overlay_dir


def use_assets_dir(assets_dir: Path) -> None:
    """Point LIBERO at an assets directory. Call before constructing any `LiberoTask`.

    LIBERO caches its assets path in a module global after the first lookup
    (`libero.libero._assets_path_cache`); every object XML is loaded from it at
    env construction, so overwriting it is enough, and it survives env resets.
    """
    import libero.libero as libero_pkg

    libero_pkg._assets_path_cache = str(assets_dir)


def copy_object_folder(original_assets: Path, object_folder: str, dst: Path) -> Path:
    """Copy one scanned object's folder (small) to `dst`, emptied first; returns the XML path inside it."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(original_assets / "stable_scanned_objects" / object_folder, dst, symlinks=False)
    return dst / f"{object_folder}.xml"
