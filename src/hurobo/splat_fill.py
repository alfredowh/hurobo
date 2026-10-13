"""Complete a scanned flat dish splat with a measured shape prior: drop off-surface haze, fill the gaps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814  # zeroth SH basis constant: colour = f_dc * C0 + 0.5


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


@dataclass(frozen=True)
class DishPrior:
    """A rotationally symmetric shallow dish (plate) on the floor: a flat base and a rising rim.

    Attributes:
        diameter_m: outer diameter, measured.
        height_m: total height from the floor to the rim top, measured.
        flat_fraction: share of the radius that is flat base; an assumption, not measured.
        thickness_m: material thickness; an assumption, not measured.
    """

    diameter_m: float
    height_m: float
    flat_fraction: float = 0.55
    thickness_m: float = 0.006

    def top_height(self, r_over_radius: np.ndarray) -> np.ndarray:
        """Height of the top surface above the floor, in metres, at r / outer radius."""
        rise = np.clip((r_over_radius - self.flat_fraction) / (1.0 - self.flat_fraction), 0.0, 1.0)
        return self.thickness_m + (self.height_m - self.thickness_m) * rise

    def top_slope(self, r_over_radius: np.ndarray) -> np.ndarray:
        """d(height)/d(r) of the top surface (dimensionless), zero on the flat base."""
        radius = self.diameter_m / 2.0
        inside_rim = (r_over_radius > self.flat_fraction) & (r_over_radius < 1.0)
        return np.where(
            inside_rim, (self.height_m - self.thickness_m) / ((1.0 - self.flat_fraction) * radius), 0.0
        )


@dataclass
class FillReport:
    """What `DishFiller.fill` did.

    Attributes:
        kept: scanned Gaussians kept.
        dropped: scanned Gaussians removed as off-surface.
        added: synthetic Gaussians inserted into gaps.
        units_per_metre: splat units per metre, from the footprint radius against the measured diameter.
        radius_units: fitted outer radius in splat units.
    """

    kept: int
    dropped: int
    added: int
    units_per_metre: float
    radius_units: float


class DishFiller:
    """Fit a `DishPrior` to an object splat (z-up, base at z = 0) and complete it along that surface."""

    def __init__(
        self,
        prior: DishPrior,
        surface_tolerance_m: float = 0.008,
        gap_spacing_factor: float = 1.5,
        added_opacity: float = 0.9,
    ) -> None:
        """Configure the fill.

        Args:
            prior: measured dish shape.
            surface_tolerance_m: scanned Gaussians further than this from the dish's top surface
                (above it, or below it by more than the thickness) are dropped as haze.
            gap_spacing_factor: a surface point counts as a gap when no kept Gaussian is within this
                many median spacings; synthetic Gaussians go only there.
            added_opacity: opacity of synthetic Gaussians.
        """
        self._prior = prior
        self._tolerance_m = surface_tolerance_m
        self._gap_factor = gap_spacing_factor
        self._added_opacity = added_opacity

    def fill(self, ply_in: Path, ply_out: Path) -> FillReport:
        """Write the completed splat.

        Args:
            ply_in: object splat from `extract_object_splat`, in SfM units.
            ply_out: destination .ply, same units and frame.

        Returns:
            Counts and the fitted scale.

        Raises:
            ValueError: if fewer than 50 solid Gaussians are found.
        """
        vertices = PlyData.read(str(ply_in))["vertex"].data.copy()
        xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
        opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
        solid = opacity > 0.3
        if solid.sum() < 50:
            raise ValueError("fewer than 50 solid Gaussians, nothing to fit the dish to")

        centre_xy = np.median(xyz[solid, :2], axis=0)
        radial = np.linalg.norm(xyz[:, :2] - centre_xy, axis=1)
        radius_units = self._rim_radius(xyz[solid, :2] - centre_xy)
        units_per_metre = radius_units / (self._prior.diameter_m / 2.0)

        r_ratio = radial / radius_units
        surface_z = self._prior.top_height(np.clip(r_ratio, 0.0, 1.0)) * units_per_metre
        tolerance = self._tolerance_m * units_per_metre
        thickness = self._prior.thickness_m * units_per_metre
        near_surface = (
            (xyz[:, 2] <= surface_z + tolerance)
            & (xyz[:, 2] >= surface_z - thickness - tolerance)
            & (r_ratio <= 1.08)
        )
        kept = vertices[near_surface]
        kept_xyz = xyz[near_surface]

        # Only solid Gaussians count as coverage: a translucent patch is the hole being filled.
        kept_solid = kept_xyz[opacity[near_surface] > 0.3]
        # Lattice spacing: three median neighbour distances, so the fill adds tens of thousands of
        # Gaussians, not as many as the scan has.
        spacing = 3.0 * float(np.median(cKDTree(kept_solid).query(kept_solid, k=2)[0][:, 1]))
        grid = self._surface_grid(centre_xy, radius_units, units_per_metre, spacing)
        distance, _ = cKDTree(kept_solid).query(grid, k=1)
        gaps = distance > self._gap_factor * spacing / 3.0
        added = self._synthetic_gaussians(
            grid[gaps], kept, kept_xyz, centre_xy, radius_units, spacing, vertices.dtype
        )

        merged = np.concatenate([kept, added]) if len(added) else kept
        ply_out.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(merged, "vertex")]).write(str(ply_out))
        return FillReport(
            kept=int(len(kept)),
            dropped=int((~near_surface).sum()),
            added=int(len(added)),
            units_per_metre=units_per_metre,
            radius_units=radius_units,
        )

    @staticmethod
    def _rim_radius(centred_xy: np.ndarray, sectors: int = 36) -> float:
        """Rim radius: median over azimuth sectors of each sector's 95th-percentile radius.

        A warped scan has a squarish outline; the sector median follows the plate's own size and ignores
        corners and single-sided haze, which a global percentile of the radius would follow.
        """
        azimuth = np.arctan2(centred_xy[:, 1], centred_xy[:, 0])
        sector = ((azimuth + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
        radius = np.linalg.norm(centred_xy, axis=1)
        per_sector = [np.percentile(radius[sector == k], 95) for k in range(sectors) if (sector == k).any()]
        return float(np.median(per_sector))

    def _surface_grid(
        self, centre_xy: np.ndarray, radius_units: float, units_per_metre: float, spacing: float
    ) -> np.ndarray:
        """Points on the dish's top surface in a square lattice of `spacing`, shape (G, 3)."""
        axis = np.arange(-radius_units, radius_units + spacing, spacing)
        gx, gy = np.meshgrid(axis, axis)
        r_ratio = np.hypot(gx, gy) / radius_units
        inside = r_ratio <= 1.0
        z = self._prior.top_height(r_ratio[inside]) * units_per_metre
        return np.c_[gx[inside] + centre_xy[0], gy[inside] + centre_xy[1], z]

    def _synthetic_gaussians(
        self,
        points: np.ndarray,
        kept: np.ndarray,
        kept_xyz: np.ndarray,
        centre_xy: np.ndarray,
        radius_units: float,
        spacing: float,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Flat surface-aligned Gaussians at `points`, coloured by the nearest kept Gaussians."""
        out = np.zeros(len(points), dtype=dtype)
        if len(points) == 0:
            return out
        colours = np.stack([kept["f_dc_0"], kept["f_dc_1"], kept["f_dc_2"]], axis=1)
        rgb = np.clip(colours * SH_C0 + 0.5, 0.0, 1.0)
        # Only bright, low-saturation Gaussians are the plate's glaze: floor haze is brown and the printed
        # pattern is dark, and neither should be spread into the gaps.
        glaze = (rgb.max(axis=1) > 0.6) & ((rgb.max(axis=1) - rgb.min(axis=1)) < 0.25)
        glaze &= 1.0 / (1.0 + np.exp(-kept["opacity"])) > 0.3
        if glaze.sum() < 10:
            raise ValueError("fewer than 10 bright low-saturation Gaussians to take the plate colour from")
        _, neighbours = cKDTree(kept_xyz[glaze]).query(points, k=8)
        out_colour = colours[glaze][neighbours].mean(axis=1)
        out["x"], out["y"], out["z"] = points[:, 0], points[:, 1], points[:, 2]
        out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = out_colour.T
        out["opacity"] = _logit(self._added_opacity)
        # Flat discs: in-plane extent matches the lattice spacing, the normal axis is thin.
        out["scale_0"] = out["scale_1"] = np.log(0.7 * spacing)
        out["scale_2"] = np.log(0.15 * spacing)
        radial = points[:, :2] - centre_xy
        r_ratio = np.linalg.norm(radial, axis=1) / radius_units
        slope = self._prior.top_slope(r_ratio)
        out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = self._normal_quaternions(radial, slope).T
        return out

    @staticmethod
    def _normal_quaternions(radial_xy: np.ndarray, slope: np.ndarray) -> np.ndarray:
        """wxyz quaternions rotating +z onto the surface normal of a surface of revolution.

        The normal of z = f(r) is (-f' * r_hat, 1) normalised, so +z tilts away from the axis by atan(f').
        """
        tilt = np.arctan(slope)
        norm = np.linalg.norm(radial_xy, axis=1)
        r_hat = np.where(norm[:, None] > 1e-9, radial_xy / np.maximum(norm[:, None], 1e-9), [1.0, 0.0])
        # Rotation axis = z x (-r_hat) = perpendicular to the tilt direction, in the xy plane.
        axis = np.c_[r_hat[:, 1], -r_hat[:, 0], np.zeros(len(r_hat))]
        half = tilt / 2.0
        return np.c_[np.cos(half), axis * np.sin(half)[:, None]]


def solid_coverage(ply_path: Path, radius_units: float, cell_fraction: float = 0.05) -> float:
    """Share of the dish's footprint covered by solid Gaussians, seen from above.

    A proxy for 'no holes': each top-down cell accumulates 1 - prod(1 - opacity) over the Gaussians
    centred in it, and counts as covered at 0.8. Gaussian extent is ignored, so it undercounts gaps
    that a large neighbour would cover in a render; compare splats with it, do not read it as an alpha.

    Args:
        ply_path: object splat, z-up, centred on the origin horizontally.
        radius_units: footprint radius to evaluate, in the splat's units.
        cell_fraction: cell edge as a fraction of the radius.

    Returns:
        Fraction of cells inside the footprint circle that are covered, in [0, 1].
    """
    vertices = PlyData.read(str(ply_path))["vertex"].data
    xy = np.stack([vertices["x"], vertices["y"]], axis=1)
    centre = np.median(xy, axis=0)
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
    cell = cell_fraction * radius_units
    bins = int(np.ceil(2 * radius_units / cell))
    index = np.floor((xy - centre + radius_units) / cell).astype(int)
    valid = ((index >= 0) & (index < bins)).all(axis=1)
    log_clear = np.zeros((bins, bins))
    np.add.at(log_clear, (index[valid, 0], index[valid, 1]), np.log1p(-np.clip(opacity[valid], 0, 0.999)))
    alpha = 1.0 - np.exp(log_clear)
    centres = (np.arange(bins) + 0.5) * cell - radius_units
    inside = np.hypot(*np.meshgrid(centres, centres, indexing="ij")) <= radius_units
    return float((alpha[inside] >= 0.8).mean())
