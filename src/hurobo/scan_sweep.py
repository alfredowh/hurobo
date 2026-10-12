"""Parameter sweep over the scan pipeline: named configs on one clip, with reloadable results."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import pandas as pd
import torch

from hurobo.scan import (
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


@dataclass(frozen=True)
class ScanConfig:
    """One point of the sweep: how to sample, reconstruct and train.

    Attributes:
        name: folder and table label, e.g. "lowthr_exhaustive".
        sfm: COLMAP settings.
        num_frames: keyframes sampled from the clip (spacing = duration / num_frames).
        iterations: splat optimisation steps.
    """

    name: str
    sfm: SfmOptions = field(default_factory=SfmOptions)
    num_frames: int = 120
    iterations: int = 7000


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

    def load_sfm(self) -> SfmResult:
        """The config's largest COLMAP model as poses and points (no pixels), in its SfM frame."""
        return sfm_result_from_model(largest_model(load_sparse_models(self.sparse_dir)))


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

    def __init__(self, video_path: Path, out_dir: Path, holdout_every: int = 10) -> None:
        """Args:
        video_path: the orbit clip.
        out_dir: sweep root, created if missing.
        holdout_every: withhold every N-th keyframe to score each splat on unseen views.
        """
        self._video = video_path
        self._out = out_dir
        self._holdout_every = holdout_every
        self._duration_s = video_duration_s(video_path)

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
        trainer = SplatTrainer(iterations=config.iterations, background_rgb=border_colour(images_rgb))
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
            KeyframeSampler(num_frames=config.num_frames).sample(self._video, images_dir)
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
