"""Parameter sweep over the scan pipeline: named configs on one clip, with reloadable results."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import pandas as pd
import torch

from hurobo.scan import (
    CropCylinder,
    KeyframeSampler,
    SfmOptions,
    SfmReconstructor,
    SfmResult,
    SfmSummary,
    SplatTrainer,
    border_colour,
    largest_model,
    load_sparse_models,
    object_frame_from_orbit,
    orbit_coverage_deg,
    preview_grid,
    sfm_result_from_model,
)
from hurobo.scan_crop import ObjectSplat, choose_object_crop, extract_object_splat, locate_white_object


@dataclass(frozen=True)
class ScanConfig:
    """One point of the sweep: how to sample, reconstruct and train.

    Attributes:
        name: folder and table label, e.g. "lowthr_exhaustive".
        sfm: COLMAP settings.
        num_frames: keyframes sampled from the clip (spacing = duration / num_frames).
        iterations: splat optimisation steps.
        grow_grad2d: densification gradient threshold (see `SplatTrainer`).
        reset_every: opacity-reset interval in steps (see `SplatTrainer`).
    """

    name: str
    sfm: SfmOptions = field(default_factory=SfmOptions)
    num_frames: int = 120
    iterations: int = 7000
    grow_grad2d: float = 2e-4
    reset_every: int = 3000


@dataclass
class SweepResult:
    """Stored outcome of one `ScanConfig`.

    Attributes:
        config: what was run.
        sfm_summary: registration statistics of the largest COLMAP model.
        orbit_coverage_deg: degrees of the orbit the registered cameras cover. Read it next to the
            PSNRs: a fragment of an arc can score a high PSNR on its own few views.
        keyframe_spacing_s: seconds between consecutive keyframes.
        num_gaussians: Gaussians in the trained splat.
        train_psnr_db: PSNR over training views (optimistic).
        heldout_psnr_db: PSNR over withheld views, or None when nothing was withheld.
        sfm_seconds: wall time of the SfM run (as measured when it first ran).
        train_seconds: wall time of splat training.
        sparse_dir: COLMAP `sparse/` folder of this config's reconstruction.
        ply_path: trained splat, in the object frame, uncropped.
        preview_path: PNG of `photo | render` rows for withheld views (training views if none).
    """

    config: ScanConfig
    sfm_summary: SfmSummary
    orbit_coverage_deg: float
    keyframe_spacing_s: float
    num_gaussians: int
    train_psnr_db: float
    heldout_psnr_db: float | None
    sfm_seconds: float
    train_seconds: float
    sparse_dir: Path
    ply_path: Path
    preview_path: Path

    def load_sfm(self, with_pixels: bool = False) -> SfmResult:
        """The config's largest COLMAP model in its SfM frame.

        Args:
            with_pixels: also load the keyframe images (needed to triangulate from image
                content); otherwise only poses and points, which is fast.
        """
        model = largest_model(load_sparse_models(self.sparse_dir))
        if not with_pixels:
            return sfm_result_from_model(model)
        # sparse_dir is <sweep>/_sfm/<key>/sparse; the keyframes live in <sweep>/_frames/<N>f.
        images_dir = self.sparse_dir.parents[2] / "_frames" / f"{self.config.num_frames}f"
        return sfm_result_from_model(model, images_dir)


def default_bowl_sweep() -> list[ScanConfig]:
    """The configs used to investigate why a bowl on a wooden floor reconstructed badly.

    Returns:
        COLMAP's defaults first, then one knob at a time (SIFT threshold, matching mode), the
        combination that fixed it, and two variations of that: sparser keyframes, longer training.
    """
    robust = SfmOptions(
        peak_threshold=0.002, max_ratio=0.9, matching="exhaustive", init_min_tri_angle_deg=4.0
    )
    return [
        ScanConfig("colmap_defaults"),
        ScanConfig("low_sift_threshold", SfmOptions(peak_threshold=0.002, max_ratio=0.9)),
        ScanConfig("exhaustive_matching", SfmOptions(matching="exhaustive")),
        ScanConfig("low_threshold_exhaustive", robust),
        ScanConfig("sparse_keyframes_40", robust, num_frames=40),
        ScanConfig("train_30k", robust, iterations=30000),
    ]


def default_clip_sweep() -> list[ScanConfig]:
    """Configs for one clip of a new object: COLMAP's defaults, then the robust settings varied.

    Returns:
        defaults; robust SIFT with sequential matching; robust + exhaustive matching at 120 frames,
        and at 60 frames (the budget the multi-clip scans use per clip).
    """
    robust = SfmOptions.robust()
    exhaustive = SfmOptions.robust_exhaustive()
    return [
        ScanConfig("colmap_defaults"),
        ScanConfig("robust_sequential", robust),
        ScanConfig("robust_exhaustive", exhaustive),
        ScanConfig("robust_exhaustive_60f", exhaustive, num_frames=60),
    ]


def plate_fill_sweep() -> list[ScanConfig]:
    """Training settings aimed at filling the weakly textured plate: robust SfM, one change at a time.

    Returns:
        longer training; lower densification threshold; opacity resets off; all three together.
    """
    exhaustive = SfmOptions.robust_exhaustive()
    return [
        ScanConfig("train_30k", exhaustive, iterations=30000),
        ScanConfig("grad1e-4", exhaustive, grow_grad2d=1e-4),
        ScanConfig("no_reset", exhaustive, reset_every=10**9),
        ScanConfig("fill_all", exhaustive, iterations=30000, grow_grad2d=1e-4, reset_every=10**9),
    ]


def combined_clip_sweep(num_clips: int, frames_per_clip: int = 60) -> list[ScanConfig]:
    """The config for several clips scanned together: exhaustive matching, since sequential links
    across clips are meaningless, at the same keyframes-per-clip budget as `robust_exhaustive_60f`.

    Args:
        num_clips: how many clips go into the one reconstruction.
        frames_per_clip: keyframe budget per clip; the total grows with the clip count.
    """
    return [ScanConfig("combined", SfmOptions.robust_exhaustive(), num_frames=frames_per_clip * num_clips)]


def video_duration_s(video_path: Path) -> float:
    """Clip length in seconds from the container's frame count and rate."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()


def sweep_table(results: list[SweepResult]) -> pd.DataFrame:
    """One row per config with the columns that matter when comparing scans.

    Args:
        results: from `ScanSweep.run` or `ScanSweep.load`.
    """
    rows = []
    for r in results:
        s = r.sfm_summary
        rows.append(
            {
                "config": r.config.name,
                "matching": r.config.sfm.matching,
                "SIFT threshold": r.config.sfm.peak_threshold or "default",
                "frames": r.config.num_frames,
                "spacing (s)": round(r.keyframe_spacing_s, 2),
                "iterations": r.config.iterations,
                "registered": f"{s.num_registered}/{s.num_images}",
                "registered %": round(100 * s.registered_fraction),
                "orbit covered (deg)": round(r.orbit_coverage_deg),
                "models": s.num_models,
                "SfM points": s.num_points,
                "reproj (px)": round(s.mean_reprojection_px, 2),
                "train PSNR (dB)": round(r.train_psnr_db, 1),
                "held-out PSNR (dB)": None if r.heldout_psnr_db is None else round(r.heldout_psnr_db, 1),
                "SfM (s)": round(r.sfm_seconds),
                "train (s)": round(r.train_seconds),
            }
        )
    return pd.DataFrame(rows)


class ScanSweep:
    """Runs `ScanConfig`s on one video and stores each result under `out_dir/<config name>/`.

    SfM is the slow part and depends only on (frame count, SfM options), so it is cached
    under `out_dir/_sfm/` and shared between configs that differ only in training.
    """

    def __init__(self, video_path: Path | list[Path], out_dir: Path, holdout_every: int = 10) -> None:
        """Args:
        video_path: the orbit clip, or several clips of one scene to reconstruct together.
        out_dir: sweep root, created if missing.
        holdout_every: withhold every N-th keyframe to score each splat on unseen views.
        """
        self._videos = [video_path] if isinstance(video_path, Path) else list(video_path)
        self._out = out_dir
        self._holdout_every = holdout_every
        self._duration_s = sum(video_duration_s(path) for path in self._videos)

    def run(self, configs: list[ScanConfig]) -> list[SweepResult]:
        """Run every config in order and write `result.json` per config, adding them to `sweep.json`.

        Args:
            configs: unique names; each is (re)trained, SfM is reused when cached.

        Returns:
            One `SweepResult` per config, in order.
        """
        self._out.mkdir(parents=True, exist_ok=True)
        results = [self._run_one(config) for config in configs]
        index_path = self._out / "sweep.json"
        known = json.loads(index_path.read_text()) if index_path.exists() else []
        index_path.write_text(json.dumps(known + [c.name for c in configs if c.name not in known]))
        return results

    @classmethod
    def load(cls, out_dir: Path) -> list[SweepResult]:
        """Reload the results a previous `run` stored, in the order it ran them."""
        names = json.loads((out_dir / "sweep.json").read_text())
        return [_result_from_json(out_dir, name) for name in names]

    def _run_one(self, config: ScanConfig) -> SweepResult:
        """SfM (cached) then training for one config; persist and return the result."""
        sfm, summary, sfm_seconds, sparse_dir = self._sfm(config)
        images_rgb = [im.image_rgb for im in sfm.images]
        torch.manual_seed(0)  # unseeded quaternion init would otherwise blur config-to-config differences
        trainer = SplatTrainer(
            iterations=config.iterations,
            background_rgb=border_colour(images_rgb),
            grow_grad2d=config.grow_grad2d,
            reset_every=config.reset_every,
        )
        config_dir = self._out / config.name
        config_dir.mkdir(parents=True, exist_ok=True)
        ply_path = config_dir / "splat.ply"
        start = time.perf_counter()
        train = trainer.train(sfm, object_frame_from_orbit(sfm), ply_path, holdout_every=self._holdout_every)
        train_seconds = time.perf_counter() - start

        preview_path = config_dir / "heldout_preview.png"
        shown = train.heldout_previews or train.previews
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview_grid(shown), cv2.COLOR_RGB2BGR))
        result = SweepResult(
            config=config,
            sfm_summary=summary,
            orbit_coverage_deg=orbit_coverage_deg(sfm),
            keyframe_spacing_s=self._duration_s / config.num_frames,
            num_gaussians=train.num_gaussians,
            train_psnr_db=train.mean_psnr_db,
            heldout_psnr_db=train.heldout_psnr_db,
            sfm_seconds=sfm_seconds,
            train_seconds=train_seconds,
            sparse_dir=sparse_dir.relative_to(self._out),
            ply_path=ply_path.relative_to(self._out),
            preview_path=preview_path.relative_to(self._out),
        )
        (config_dir / "result.json").write_text(json.dumps(_result_to_json(result), indent=2))
        print(
            f"[sweep] {config.name}: registered {summary.num_registered}/{summary.num_images}, "
            f"held-out {train.heldout_psnr_db}"
        )
        return result

    def _sfm(self, config: ScanConfig) -> tuple[SfmResult, SfmSummary, float, Path]:
        """Keyframes + SfM for a config, reusing `_sfm/<key>/` when an identical run exists."""
        key_source = json.dumps([asdict(config.sfm), config.num_frames], sort_keys=True)
        key = f"{config.num_frames}f_{hashlib.md5(key_source.encode()).hexdigest()[:8]}"
        work = self._out / "_sfm" / key
        meta_path = work / "summary.json"
        images_dir = self._out / "_frames" / f"{config.num_frames}f"
        if not images_dir.exists():
            KeyframeSampler(num_frames=config.num_frames).sample_many(self._videos, images_dir)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            summary, seconds = SfmSummary(**meta["summary"]), meta["seconds"]
            model = largest_model(load_sparse_models(work / "sparse"))
            return sfm_result_from_model(model, images_dir), summary, seconds, work / "sparse"
        reconstructor = SfmReconstructor(config.sfm)
        start = time.perf_counter()
        sfm = reconstructor.run(images_dir, work)
        seconds = time.perf_counter() - start
        (work / "database.db").unlink(missing_ok=True)  # the match tables are large and not needed again
        meta_path.write_text(json.dumps({"summary": asdict(reconstructor.last_summary), "seconds": seconds}))
        return sfm, reconstructor.last_summary, seconds, work / "sparse"


def load_labelled(sweeps: dict[str, Path]) -> list[SweepResult]:
    """Results of several sweep folders in one list, each config renamed `<label>/<config>`.

    Args:
        sweeps: label (e.g. the clip name) -> sweep folder written by `ScanSweep.run`.
    """
    return [
        replace(result, config=replace(result.config, name=f"{label}/{result.config.name}"))
        for label, folder in sweeps.items()
        for result in ScanSweep.load(folder)
    ]


def extract_result_object(
    result: SweepResult,
    ply_out: Path,
    crop: CropCylinder | None = None,
    height_sigmas: float = 3.0,
) -> ObjectSplat:
    """The object alone (floor removed, upright, base at z = 0) from one sweep result's uncropped splat.

    Uses the pipeline's own crop choice (`choose_object_crop`) on the result's SfM points.

    Args:
        result: a stored sweep result.
        ply_out: where to write the object splat.
        crop: explicit crop cylinder in the object frame; the floor is still fitted automatically.
        height_sigmas: floor-scatter multiples to cut at; lower it for a thin object such as a plate.

    Raises:
        ValueError: if no floor plane was found or nothing stands above it.
    """
    sfm = result.load_sfm()
    in_object_frame = sfm.in_frame(object_frame_from_orbit(sfm))
    choice = choose_object_crop(
        in_object_frame.points_world, in_object_frame.camera_centres_world, explicit=crop
    )
    if choice.floor is None:
        raise ValueError("no floor plane found, cannot cut the object off it")
    ply_out.parent.mkdir(parents=True, exist_ok=True)
    return extract_object_splat(
        result.ply_path, ply_out, choice.floor, choice.cylinder, height_sigmas=height_sigmas
    )


def extract_result_objects(
    results: list[SweepResult], out_dir: Path, min_registered_fraction: float = 0.8
) -> tuple[dict[str, Path], dict[str, str]]:
    """`extract_result_object` for every result that registered enough, with failures reported, not raised.

    Args:
        results: from `load_labelled` or `ScanSweep.load`.
        out_dir: object splats go to `<out_dir>/<config name>.ply`.
        min_registered_fraction: results below this registered share are skipped (a fragment of an
            orbit is not an object scan).

    Returns:
        (config name -> object splat path, config name -> why it was skipped or failed).
    """
    objects: dict[str, Path] = {}
    skipped: dict[str, str] = {}
    for result in results:
        name = result.config.name
        fraction = result.sfm_summary.registered_fraction
        if fraction < min_registered_fraction:
            skipped[name] = f"only {100 * fraction:.0f} % of keyframes registered"
            continue
        ply_out = out_dir / (name.replace("/", "__") + ".ply")
        try:
            extract_result_object(result, ply_out)
        except ValueError as error:
            skipped[name] = str(error)
            continue
        objects[name] = ply_out
    return objects, skipped


def white_object_residuals(results: list[SweepResult]) -> pd.DataFrame:
    """How consistently each reconstruction places the one white object across its keyframes.

    Triangulates the white pixels (`locate_white_object`); a small mean ray distance means the
    rays from all views meet, i.e. the cameras are mutually consistent around the object. It
    is an independent check on SfM that does not rely on the splat. Needs a white object that is
    the only white thing in view, so it is meaningful for the bowl clips, not the plate.

    Args:
        results: stored sweep results; each loads its keyframes, so pass the ones you care about.

    Returns:
        One row per result: object radius, mean ray distance (both SfM units) and views used.
    """
    rows = []
    for result in results:
        sfm = result.load_sfm(with_pixels=True)
        reference = locate_white_object(sfm.in_frame(object_frame_from_orbit(sfm)))
        rows.append(
            {
                "config": result.config.name,
                "white-object radius": round(reference.radius, 2),
                "mean ray distance": round(reference.mean_ray_distance, 2),
                "views used": f"{reference.num_views}/{len(sfm.images)}",
            }
        )
    return pd.DataFrame(rows)


def _result_to_json(result: SweepResult) -> dict:
    """JSON-safe dict of a result (paths as strings)."""
    data = asdict(result)
    for key in ("sparse_dir", "ply_path", "preview_path"):
        data[key] = str(data[key])
    return data


def _result_from_json(out_dir: Path, name: str) -> SweepResult:
    """Rebuild a `SweepResult` from `<out_dir>/<name>/result.json`; paths are made absolute."""
    data = json.loads((out_dir / name / "result.json").read_text())
    config = data["config"]
    coverage = data.get("orbit_coverage_deg")
    if coverage is None:
        sfm = sfm_result_from_model(largest_model(load_sparse_models(out_dir / data["sparse_dir"])))
        coverage = orbit_coverage_deg(sfm)
    return SweepResult(
        config=ScanConfig(
            name=config["name"],
            sfm=SfmOptions(**config["sfm"]),
            num_frames=config["num_frames"],
            iterations=config["iterations"],
            grow_grad2d=config.get("grow_grad2d", 2e-4),
            reset_every=config.get("reset_every", 3000),
        ),
        sfm_summary=SfmSummary(**data["sfm_summary"]),
        orbit_coverage_deg=coverage,
        keyframe_spacing_s=data["keyframe_spacing_s"],
        num_gaussians=data["num_gaussians"],
        train_psnr_db=data["train_psnr_db"],
        heldout_psnr_db=data["heldout_psnr_db"],
        sfm_seconds=data["sfm_seconds"],
        train_seconds=data["train_seconds"],
        sparse_dir=out_dir / data["sparse_dir"],
        ply_path=out_dir / data["ply_path"],
        preview_path=out_dir / data["preview_path"],
    )
