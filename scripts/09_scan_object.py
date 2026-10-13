"""Orbit video -> 3DGS splat .ply(s) for `08_splat_composite.py` (photoreal stretch goal, PLAN.md §5).

Two modes:
- default: one object alone (turntable or orbit), no board. Frame from the orbit
  geometry (z up, base at z=0), arbitrary scale. Writes `<name>_full.ply` (everything)
  and `<name>.ply` (cropped to the object). When a floor plane was found it also writes
  `<name>_object.ply`: the object alone, floor removed, upright, base at z=0, ready to attach.
  The crop is, in order: `--crop X Y R` if given; the floor-plane + cluster estimate (a bowl
  standing on a floor); the old percentile-of-all-points fit (a turntable on a plain backdrop).
  Attach with MATCH_BODY_XY.
- --charuco: the real table with the ChArUco board lying on it (REAL2SIM_IDEAS.md
  §7 capture). Metric, in the board's z-up table frame. Each `--object NAME X Y R H`
  (metres, table frame) is cut out to `<name>_<NAME>.ply` and removed from
  `<name>_table.ply`. With `--table-radius` the table splat is split into a
  tabletop cylinder (`<name>_table.ply`) and `<name>_background.ply`.
  Attach objects with METRIC_BOTTOM, the table with METRIC_TOP.

SfM uses the robust settings (lowered SIFT threshold) and retries with exhaustive matching
when under 80 % of the keyframes register (design log D-001). `--holdout-every N` also
reports PSNR on views withheld from training.

    uv run python scripts/09_scan_object.py data/raw/dummy_orbits/<clip>.mp4 --name dummy_fruit_bowl
    uv run python scripts/09_scan_object.py data/raw/phone_bowl.mp4 --name bowl --holdout-every 10
    uv run python scripts/09_scan_object.py data/raw/phone_bowl.mp4 --name bowl --recrop --crop 0.1 0.5 2.7
    uv run python scripts/09_scan_object.py data/raw/s24_orbit.mp4 --name real --charuco \\
        --object bowl 0.10 0.05 0.09 0.08 --object plate -0.05 0.12 0.12 0.04 \\
        --table-radius 0.5 --table-centre 0.2 0.1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from hurobo.camera import ChArUcoCalibrator
from hurobo.scan import (
    CropCylinder,
    KeyframeSampler,
    SfmReconstructor,
    SfmResult,
    Similarity,
    SplatTrainer,
    TrainResult,
    align_charuco_table,
    border_colour,
    crop_ply,
    largest_model,
    load_sparse_models,
    object_frame_from_orbit,
    preview_grid,
    sfm_result_from_model,
    split_tabletop_background,
    transform_ply,
)
from hurobo.scan_crop import (
    ExpectedObject,
    choose_object_crop,
    extract_object_splat,
    locate_objects_on_table,
)


def save_preview(result: TrainResult, path: Path) -> None:
    """Write a photo | splat grid of a few held-out views, or training views if none were withheld."""
    shown = result.heldout_previews or result.previews
    cv2.imwrite(str(path), cv2.cvtColor(preview_grid(shown), cv2.COLOR_RGB2BGR))


def scan_single_object(
    sfm: SfmResult,
    trainer: SplatTrainer,
    out_dir: Path,
    name: str,
    explicit_crop: CropCylinder | None,
    holdout_every: int | None,
) -> TrainResult:
    """Orbit-frame mode: train once in the object frame (uncropped), then crop to the object."""
    t_obj_world = object_frame_from_orbit(sfm)
    full_ply = out_dir / f"{name}_full.ply"
    result = trainer.train(sfm, t_obj_world, full_ply, holdout_every=holdout_every)
    crop_single_object(sfm, full_ply, out_dir / f"{name}.ply", explicit_crop)
    return result


def crop_single_object(
    sfm: SfmResult, full_ply: Path, out_ply: Path, explicit_crop: CropCylinder | None
) -> CropCylinder:
    """Crop an uncropped object-frame splat to the object, and cut the object off the floor if one was found.

    Writes `out_ply` (the crop, floor included) and, when a floor plane was fitted,
    `<stem>_object.ply`: only the object, upright with its base at z = 0.
    """
    t_obj_world = object_frame_from_orbit(sfm)
    in_object_frame = sfm.in_frame(t_obj_world)
    choice = choose_object_crop(
        in_object_frame.points_world, in_object_frame.camera_centres_world, explicit_crop
    )
    cylinder = choice.cylinder
    kept = crop_ply(full_ply, out_ply, [cylinder])
    print(
        f"[scan] crop ({choice.method}): centre ({cylinder.centre_xy[0]:.2f}, {cylinder.centre_xy[1]:.2f}), "
        f"radius {cylinder.radius:.2f}, z {cylinder.z_min:.2f}..{cylinder.z_max:.2f} -> "
        f"{kept} Gaussians in {out_ply}"
    )
    if choice.floor is None:
        print("[scan] no floor plane found: skipped the object-only splat")
        return cylinder
    object_ply = out_ply.with_name(f"{out_ply.stem}_object.ply")
    info = extract_object_splat(full_ply, object_ply, choice.floor, cylinder)
    print(
        f"[scan] object only: {info.num_gaussians} Gaussians, radius {info.footprint_radius:.2f}, "
        f"height {info.height:.2f}, floor tilt {info.tilt_removed_deg:.1f} deg removed -> {object_ply}"
    )
    return cylinder


def align_charuco(sfm: SfmResult, board: ChArUcoCalibrator, board_thickness_m: float) -> Similarity:
    """SfM frame -> metric table frame (see `align_charuco_table`), printing the corner residual."""
    similarity, rms_m = align_charuco_table(sfm, board.board, board_thickness_m)
    print(f"[scan] ChArUco alignment: scale {similarity.scale:.4f} m/unit, corner RMS {1000 * rms_m:.1f} mm")
    return similarity


def cut_charuco_table(
    similarity: Similarity,
    raw_ply: Path,
    out_dir: Path,
    name: str,
    objects: list[list[str]],
    table_radius_m: float | None = None,
    table_centre_xy: tuple[float, float] = (0.0, 0.0),
    table_box_xy: tuple[float, float, float, float] | None = None,
    find_objects: list[ExpectedObject] | None = None,
) -> None:
    """Move the raw splat to the table frame, cut objects out, split tabletop from background."""
    table_ply = out_dir / f"{name}_table.ply"
    transform_ply(raw_ply, table_ply, similarity)

    named_cylinders = {
        obj_name: CropCylinder(float(radius), -0.01, float(height), np.array([float(x), float(y)]))
        for obj_name, x, y, radius, height in objects
    }
    if find_objects:
        found = locate_objects_on_table(table_ply, find_objects, region_xy=table_box_xy)
        for obj_name, cylinder in found.items():
            print(
                f"[scan] found {obj_name} at x {cylinder.centre_xy[0]:.3f}, y {cylinder.centre_xy[1]:.3f} m "
                f"(radius {cylinder.radius:.3f}, height {cylinder.z_max:.3f})"
            )
        named_cylinders.update(found)
    cylinders = list(named_cylinders.values())
    for obj_name, cylinder in named_cylinders.items():
        kept = crop_ply(table_ply, out_dir / f"{name}_{obj_name}.ply", [cylinder])
        print(f"[scan] {obj_name}: {kept} Gaussians -> {out_dir / f'{name}_{obj_name}.ply'}")
    if cylinders:
        kept = crop_ply(table_ply, table_ply, cylinders, keep_inside=False)
        print(f"[scan] table without objects: {kept} Gaussians -> {table_ply}")
    if table_radius_m is not None or table_box_xy is not None:
        background_ply = out_dir / f"{name}_background.ply"
        tabletop, background = split_tabletop_background(
            table_ply, background_ply, table_radius_m, table_centre_xy, box_xy=table_box_xy
        )
        print(f"[scan] tabletop {tabletop} -> {table_ply}; background {background} -> {background_ply}")


def main() -> None:
    """Parse args, run keyframes -> SfM -> training -> framing/cropping; write .ply files and a preview."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "video", type=Path, nargs="+", help="one clip, or several clips of the same scene (one SfM model)"
    )
    parser.add_argument("--name", required=True, help="output stem, e.g. 'my_bowl'")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument(
        "--holdout-every",
        type=int,
        default=None,
        help="withhold every N-th keyframe from training and report PSNR on them",
    )
    parser.add_argument(
        "--charuco", action="store_true", help="metric table scan with the ChArUco board in view"
    )
    parser.add_argument(
        "--crop",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "RADIUS"),
        help="single-object mode: crop cylinder centre and radius in the object frame (SfM units); "
        "overrides the automatic crop",
    )
    parser.add_argument(
        "--recrop",
        action="store_true",
        help="skip SfM and training and redo the crop: <name>_full.ply (single object) or, with --charuco, "
        "the table-frame cut of <name>_sfm_frame.ply",
    )
    parser.add_argument(
        "--object",
        nargs=5,
        action="append",
        default=[],
        metavar=("NAME", "X", "Y", "RADIUS", "HEIGHT"),
        help="--charuco only: object cylinder to cut out, metres in the table frame",
    )
    parser.add_argument(
        "--find-object",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "DIAMETER", "HEIGHT"),
        help="--charuco only: cut out an object located automatically from its measured size, metres",
    )
    parser.add_argument(
        "--table-radius",
        type=float,
        default=None,
        help="--charuco only: tabletop radius in metres; the rest goes to <name>_background.ply",
    )
    parser.add_argument(
        "--table-centre",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help="--charuco only: table centre (x y) in the board's table frame, metres",
    )
    parser.add_argument(
        "--table-box",
        nargs=4,
        type=float,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="--charuco only: rectangular tabletop in the table frame, metres (overrides --table-radius)",
    )
    parser.add_argument(
        "--square-length", type=float, default=0.035, help="--charuco only: board square side, metres"
    )
    parser.add_argument(
        "--marker-length", type=float, default=0.026, help="--charuco only: board ArUco marker side, metres"
    )
    parser.add_argument(
        "--board-thickness",
        type=float,
        default=0.0,
        help="--charuco only: height of the board surface above the tabletop, metres (tablet thickness)",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/scans"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/splats"))
    args = parser.parse_args()
    work = args.work_dir / args.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    explicit_crop = None
    if args.crop is not None:
        x, y, radius = args.crop
        explicit_crop = CropCylinder(radius, -1e9, 1e9, np.array([x, y]))

    board = ChArUcoCalibrator(square_length_m=args.square_length, marker_length_m=args.marker_length)
    table_box_xy = tuple(args.table_box) if args.table_box else None
    find_objects = [ExpectedObject(n, float(d), float(h)) for n, d, h in args.find_object]

    if args.recrop:
        sparse = work / "colmap" / "sparse"
        full_ply = args.out_dir / f"{args.name}_{'sfm_frame' if args.charuco else 'full'}.ply"
        if not sparse.exists() or not full_ply.exists():
            parser.error(f"--recrop needs {sparse} and {full_ply} from an earlier run")
        if args.charuco:
            sfm = sfm_result_from_model(largest_model(load_sparse_models(sparse)), work / "images")
            cut_charuco_table(
                align_charuco(sfm, board, args.board_thickness),
                full_ply,
                args.out_dir,
                args.name,
                args.object,
                args.table_radius,
                tuple(args.table_centre),
                table_box_xy,
                find_objects,
            )
        else:
            sfm = sfm_result_from_model(largest_model(load_sparse_models(sparse)))
            crop_single_object(sfm, full_ply, args.out_dir / f"{args.name}.ply", explicit_crop)
        return

    t0 = time.perf_counter()
    sampler = KeyframeSampler(num_frames=args.frames)
    if len(args.video) == 1:
        sampler.sample(args.video[0], work / "images")
    else:
        sampler.sample_many(args.video, work / "images")
    reconstructor = SfmReconstructor.robust()
    sfm = reconstructor.run(work / "images", work / "colmap")
    t_sfm = time.perf_counter()

    if args.charuco:
        trainer = SplatTrainer(iterations=args.iterations, background_rgb=(0.0, 0.0, 0.0))
        similarity = align_charuco(sfm, board, args.board_thickness)  # fail before training if no board
        raw_ply = args.out_dir / f"{args.name}_sfm_frame.ply"
        result = trainer.train(sfm, np.eye(4), raw_ply, holdout_every=args.holdout_every)
        cut_charuco_table(
            similarity,
            raw_ply,
            args.out_dir,
            args.name,
            args.object,
            args.table_radius,
            tuple(args.table_centre),
            table_box_xy,
            find_objects,
        )
    else:
        trainer = SplatTrainer(
            iterations=args.iterations, background_rgb=border_colour([im.image_rgb for im in sfm.images])
        )
        result = scan_single_object(sfm, trainer, args.out_dir, args.name, explicit_crop, args.holdout_every)

    save_preview(result, work / "preview_photo_vs_splat.png")
    summary = reconstructor.last_summary
    heldout = (
        f"held-out PSNR {result.heldout_psnr_db:.1f} dB | " if result.heldout_psnr_db is not None else ""
    )
    print(
        f"registered {summary.num_registered}/{summary.num_images} keyframes in {summary.num_models} "
        f"model(s), {len(sfm.points_world)} SfM points | {heldout}"
        f"train-view PSNR {result.mean_psnr_db:.1f} dB (optimistic: training views) | "
        f"SfM {t_sfm - t0:.0f} s, training {time.perf_counter() - t_sfm:.0f} s"
    )


if __name__ == "__main__":
    main()
