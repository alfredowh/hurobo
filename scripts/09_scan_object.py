"""Orbit video -> 3DGS splat .ply(s) for `08_splat_composite.py` (photoreal stretch goal, PLAN.md §5).

Two modes:
- default: one object alone (turntable or orbit), no board. Frame from the orbit
  geometry (z up, base at z=0), arbitrary scale, cropped to the object. Attach
  with MATCH_BODY_XY. Tested on Mixkit clip 10424 (fruit bowl, white backdrop).
- --charuco: the real table with the ChArUco board lying on it (REAL2SIM_IDEAS.md
  §7 capture). Metric, in the board's z-up table frame. Each `--object NAME X Y R H`
  (metres, table frame) is cut out to `<name>_<NAME>.ply` and removed from
  `<name>_table.ply`. Attach objects with METRIC_BOTTOM, the table with METRIC_TOP.

    uv run python scripts/09_scan_object.py data/raw/dummy_orbits/<clip>.mp4 --name dummy_fruit_bowl
    uv run python scripts/09_scan_object.py data/raw/s24_orbit.mp4 --name real --charuco \\
        --object bowl 0.10 0.05 0.09 0.08 --object plate -0.05 0.12 0.12 0.04
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from hurobo.camera import ChArUcoCalibrator
from hurobo.scan import (
    ChArUcoAligner,
    CropCylinder,
    KeyframeSampler,
    SfmReconstructor,
    SfmResult,
    SplatTrainer,
    TrainResult,
    border_colour,
    crop_ply,
    object_frame_from_orbit,
    transform_ply,
)


def save_preview(result: TrainResult, path: Path) -> None:
    """Write a photo | splat grid of a few training views."""
    rows = [np.concatenate([photo, render], axis=1) for render, photo in result.previews]
    preview = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def scan_single_object(sfm: SfmResult, trainer: SplatTrainer, out_ply: Path) -> TrainResult:
    """Orbit-frame mode: train in the object frame, then crop to the SfM points' cylinder."""
    t_obj_world = object_frame_from_orbit(sfm)
    result = trainer.train(sfm, t_obj_world, out_ply)
    # Backdrop Gaussians sit outside the object; the SfM points mark where the object is.
    points_obj = sfm.points_world @ t_obj_world[:3, :3].T + t_obj_world[:3, 3]
    kept = crop_ply(out_ply, out_ply, [CropCylinder.around_points(points_obj)])
    print(f"[scan] crop kept {kept}/{result.num_gaussians} Gaussians -> {out_ply}")
    return result


def scan_charuco_table(
    sfm: SfmResult, trainer: SplatTrainer, out_dir: Path, name: str, objects: list[list[str]]
) -> TrainResult:
    """ChArUco mode: train in the SfM frame, align to the metric table frame, split objects from the table."""
    similarity, rms_m = ChArUcoAligner(ChArUcoCalibrator().board).align(sfm)
    print(f"[scan] ChArUco alignment: scale {similarity.scale:.4f} m/unit, corner RMS {1000 * rms_m:.1f} mm")
    raw_ply = out_dir / f"{name}_sfm_frame.ply"
    result = trainer.train(sfm, np.eye(4), raw_ply)
    table_ply = out_dir / f"{name}_table.ply"
    transform_ply(raw_ply, table_ply, similarity)

    cylinders = []
    for obj_name, x, y, radius, height in objects:
        cylinder = CropCylinder(float(radius), -0.01, float(height), np.array([float(x), float(y)]))
        kept = crop_ply(table_ply, out_dir / f"{name}_{obj_name}.ply", [cylinder])
        print(f"[scan] {obj_name}: {kept} Gaussians -> {out_dir / f'{name}_{obj_name}.ply'}")
        cylinders.append(cylinder)
    if cylinders:
        kept = crop_ply(table_ply, table_ply, cylinders, keep_inside=False)
        print(f"[scan] table without objects: {kept} Gaussians -> {table_ply}")
    return result


def main() -> None:
    """Parse args, run keyframes -> SfM -> training -> framing/cropping; write .ply files and a preview."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--name", required=True, help="output stem, e.g. 'my_bowl'")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument(
        "--charuco", action="store_true", help="metric table scan with the ChArUco board in view"
    )
    parser.add_argument(
        "--object",
        nargs=5,
        action="append",
        default=[],
        metavar=("NAME", "X", "Y", "RADIUS", "HEIGHT"),
        help="--charuco only: object cylinder to cut out, metres in the table frame",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/scans"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/splats"))
    args = parser.parse_args()
    work = args.work_dir / args.name
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    KeyframeSampler(num_frames=args.frames).sample(args.video, work / "images")
    sfm = SfmReconstructor().run(work / "images", work / "colmap")
    t_sfm = time.perf_counter()

    if args.charuco:
        trainer = SplatTrainer(iterations=args.iterations, background_rgb=(0.0, 0.0, 0.0))
        result = scan_charuco_table(sfm, trainer, args.out_dir, args.name, args.object)
    else:
        trainer = SplatTrainer(
            iterations=args.iterations, background_rgb=border_colour([im.image_rgb for im in sfm.images])
        )
        result = scan_single_object(sfm, trainer, args.out_dir / f"{args.name}.ply")

    save_preview(result, work / "preview_photo_vs_splat.png")
    print(
        f"registered {len(sfm.images)} views, {len(sfm.points_world)} SfM points | "
        f"train-view PSNR {result.mean_psnr_db:.1f} dB (optimistic: training views) | "
        f"SfM {t_sfm - t0:.0f} s, training {time.perf_counter() - t_sfm:.0f} s"
    )


if __name__ == "__main__":
    main()
