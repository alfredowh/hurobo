"""Build a LIBERO collision shape from a splat: depth fusion -> mesh -> CoACD -> patched asset overlay.

Writes, under --out/<name>/: fused_surface.obj, pieces/*.obj, assets/ (a LIBERO
assets tree whose `<--object>` is patched), and collider.json (rim grasp offset,
piece count, mass). `08_splat_composite.py --collider-dir` then runs the sim on it.
The splat is fitted to the sim body exactly like `SplatCompositor` does, so the
collider lies where the splat is drawn.

    uv run python scripts/10_build_collider.py --ply data/splats/dummy_discoverse/object__bowl_pink.ply \\
        --name pink_bowl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from gaussian_renderer import load_ply

from hurobo.collider import (
    GaussiansBodyFrame,
    SplatSurfaceExtractor,
    build_asset_overlay,
    copy_object_folder,
    decompose_convex,
    patch_object_xml,
    rim_grasp_offset,
)
from hurobo.sim import LiberoTask
from hurobo.splat import SplatAsset, SplatFit, body_visual_aabb, fit_gaussians_to_body


def main() -> None:
    """Parse args, fit the splat to the sim body, extract + decompose a collider, write the overlay."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fit", choices=[f.value for f in SplatFit], default=SplatFit.MATCH_BODY_XY.value)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--body", default="akita_black_bowl_1_main", help="sim body whose collider is replaced"
    )
    parser.add_argument("--object", default="akita_black_bowl", help="folder under stable_scanned_objects/")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--trunc-mm", type=float, default=5.0)
    parser.add_argument("--max-pieces", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("data/colliders"))
    args = parser.parse_args()
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)

    import mujoco

    t0 = time.perf_counter()
    task = LiberoTask(args.suite, args.task_id)
    task.reset()
    model = task.mj_model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.body)
    body_lo, body_hi = body_visual_aabb(model, body_id)
    task.close()

    asset = SplatAsset(args.body, args.ply, SplatFit(args.fit), yaw_deg=args.yaw_deg)
    raw = load_ply(str(args.ply))
    xyz, rot, scale = fit_gaussians_to_body(raw, asset, body_lo, body_hi)
    fitted = GaussiansBodyFrame(xyz, rot, scale, np.asarray(raw.opacity), np.asarray(raw.sh)[:, :3])
    extent_mm = 1000 * (
        np.percentile(xyz[raw.opacity > 0.3], 98, 0) - np.percentile(xyz[raw.opacity > 0.3], 2, 0)
    )
    print(f"[collider] splat fitted to body: extent {extent_mm.round(0)} mm")

    surface = SplatSurfaceExtractor(voxel_m=args.voxel_mm / 1000, trunc_m=args.trunc_mm / 1000).extract(
        fitted
    )
    surface.export(out / "fused_surface.obj")
    print(
        f"[collider] surface: {len(surface.vertices)} verts, {len(surface.faces)} faces, "
        f"extent {(1000 * surface.extents).round(0)} mm, watertight={surface.is_watertight}"
    )

    pieces = decompose_convex(surface, max_pieces=args.max_pieces)
    (out / "pieces").mkdir(exist_ok=True)
    for i, piece in enumerate(pieces):
        piece.export(out / "pieces" / f"piece_{i:02d}.obj")
    union = trimesh.util.concatenate(pieces)
    print(f"[collider] CoACD: {len(pieces)} convex pieces, union extent {(1000 * union.extents).round(0)} mm")

    original_assets = Path(_original_assets_dir())
    patched_dir = out / "patched_object"
    xml_in = copy_object_folder(original_assets, args.object, patched_dir)
    result = patch_object_xml(xml_in, xml_in, pieces)
    overlay = build_asset_overlay(original_assets, out / "assets", args.object, patched_dir)

    grasp = rim_grasp_offset(surface)
    (out / "collider.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "body": args.body,
                "assets_dir": str(overlay.resolve()),
                "grasp_offset_p_bowl": grasp.tolist(),
                "num_pieces": result.num_pieces,
                "replaced_geoms": result.num_removed_geoms,
                "collision_mass_kg": result.collision_mass_kg,
                "surface_extent_mm": (1000 * surface.extents).round(1).tolist(),
                "fit": args.fit,
                "yaw_deg": args.yaw_deg,
                "ply": str(args.ply.resolve()),
            },
            indent=2,
        )
    )
    print(
        f"[collider] replaced {result.num_removed_geoms} boxes with {result.num_pieces} pieces, "
        f"mass {1000 * result.collision_mass_kg:.0f} g, rim grasp offset {grasp.round(3)} -> {out}"
        f" ({time.perf_counter() - t0:.0f} s)"
    )


def _original_assets_dir() -> str:
    """LIBERO's real assets directory (before any override)."""
    from libero.libero import get_assets_path

    return get_assets_path()


if __name__ == "__main__":
    main()
