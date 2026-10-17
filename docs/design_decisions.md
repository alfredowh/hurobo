# Design decisions

What we reuse from existing work, with which settings, and where we changed it and why. Entries are numbered in the order they were made. The
rule for keeping this file current is in [CLAUDE.md](../CLAUDE.md) ("Design decisions log").

How this relates to the other documents: [PLAN.md](../PLAN.md) §2 holds the up-front research verdicts (which
framework, model and simulator to use). [README.md](../README.md) §4 lists the components. This file records what
happened **after** that: defaults that did not work on our data, and what we did about it.

## Status words

| Word | Meaning |
|---|---|
| **As-is** | Used with upstream defaults and code unchanged. |
| **Configured** | Upstream code, non-default settings. The change is in options, not in the library. |
| **Modified** | Our code changes or replaces upstream behaviour (own trainer, patched step, extra stage). |
| **Novel** | Our own method; no upstream equivalent was used. |
| **Rejected** | Evaluated and not adopted. Kept so it is not re-tried blindly. |

---

## Component register

| Component | Upstream | Status | Notes |
|---|---|---|---|
| Simulator / benchmark | LIBERO (`hf-libero`, robosuite, MuJoCo) | As-is | PLAN.md §2. |
| Policy framework | LeRobot ≥ 0.6.1 | As-is | `lerobot-train`, `lerobot-eval`, LeRobotDataset v3. |
| VLA | SmolVLA (`lerobot/smolvla_base`), ACT fallback | As-is (planned) | PLAN.md §2. |
| Hand landmarks | MediaPipe Hand Landmarker | As-is | Landmarks only. |
| Metric hand pose | MediaPipe + ArUco + `solvePnP` + anchor correction | Novel (own code, `hand.py`) | PLAN.md §2, HaWoR rejected there. |
| Clip cleaning before extraction | none (own code, `clip_clean.py`, `clip_cutter_app.py`) | Novel | Frame and clip filters, one open-close-open cycle per clip, cuts file, browser UI. D-012. |
| Marker poses, ChArUco calibration | OpenCV ArUco | As-is | |
| Structure from motion | `pycolmap` (SIFT, matching, incremental mapping) | **Configured** | D-001. Lowered SIFT threshold is the script default; exhaustive matching is the fallback. |
| Splat training | `gsplat` rasteriser and densification strategy | **Modified** | Own minimal trainer (`SplatTrainer`), see D-002, D-003. |
| Object crop | none (RANSAC plane + connected components, from `scipy`/`numpy`) | **Novel** | D-004. |
| Metric table scan (ChArUco on a tablet, tabletop box crop, object finder) | OpenCV ArUco board detection; own alignment (`ChArUcoAligner`) | **Novel** | D-009. |
| Splat rendering / compositing | DISCOVERSE `gaussian-renderer` | As-is | `load_ply` and rasterisation; compositing code is ours. |
| Collision geometry | open3d, CoACD | As-is | |
| SmolVLA checkpoints for LIBERO | `lerobot/smolvla_libero`, `HuggingFaceVLA/smolvla_libero` | As-is | D-011. Weights unchanged; our own closed-loop evaluation, since `lerobot-eval` rebuilds the stock scene. |
| SmolVLA fine-tuning | `lerobot-train` from `lerobot/smolvla_base` | Configured | D-011, D-013. Camera rename, expert-only training, batch 16, 3000 steps with a short schedule; one run on 100 generated episodes. |
| Retargeting | MimicGen idea (concept only) | Novel (own code) | MimicGen is cited, not installed. D-010. |
| LIBERO scene for E1 (scaled bowl and plate, other objects parked) | `libero_goal` task 8 | **Modified** | D-010. Our own env loop is needed; `lerobot-eval` rebuilds the stock scene. |
| Table frame from a ChArUco board for human clips | OpenCV ArUco/ChArUco | Novel (own code, `BoardMarkerPoseEstimator`) | Same frame as the table scans (D-009). D-010. |
| Demo expansion (nearest-layout and per-subtask source choice, action noise, source screening) | MimicGen (Mandlekar et al., CoRL 2023); SkillMimicGen and DexMimicGen read, not used | **Modified** (own code, `mimicgen.py`, separate from `generate.py`) | D-013. Ideas reused: nearest-neighbour and per-subtask source selection, action noise sigma 0.05, success filter. Departed: no object-pose SE(3) transform of whole segments (the retargeter's bending is kept), no hand orientation, source screening added. Not the pipeline default. |

---

## Decision log

### D-001 · 2026-10-12 · SfM settings for a low-texture phone scan

**Component:** `pycolmap`, driven by `hurobo.scan.SfmReconstructor`. **Status:** Configured.

**Symptom.** Two phone clips of a white bowl on a wooden floor (`data/raw/setup_01/`) gave bad splats. COLMAP
registered 39 of 120 keyframes in three disconnected models (1-angle clip) and 7 of 120 (3-angle clip). The splat
was trained on a thin slice of the orbit, which is also why the camera path in the viewer looked broken.

**Cause.** With COLMAP's default SIFT contrast threshold (0.0067) the images gave about 420 to 570 keypoints each: the
bowl is smooth and the floor is low-contrast in warm indoor light. 63 and 94 of 119 consecutive keyframe pairs had no
verified matches.

**Upstream defaults.** SIFT `peak_threshold` 0.0067, ratio test 0.8, sequential matching with 15 neighbours,
`init_min_tri_angle` 16°, 960 px keyframes (our sampler).

**Change.** The four settings are now fields of `hurobo.scan.SfmOptions`. The default `SfmOptions()` equals COLMAP's
defaults, so `scripts/09_scan_object.py` behaves as before. The sweep (`scripts/14_scan_param_sweep.py`) runs
non-default sets.

**Evidence** (`examples/06_scan_parameter_sweep.ipynb`, 1-angle clip, one run each, from repo code):

| Settings | Registered | Models |
|---|---|---|
| COLMAP defaults | 39/120 | 3 |
| exhaustive matching, default threshold | 53/120 | 2 |
| `peak_threshold=0.002`, ratio 0.9, sequential | **120/120** | 1 |
| same + exhaustive matching, `init_min_tri_angle=4` | 120/120 | 1 |

On the 3-angle clip sequential matching with the lower threshold stopped at 70 of 120 through the script (74 to 79
in an earlier scratch ablation), and exhaustive matching was needed for 120/120.

**Did not help** on this footage: full resolution alone (10/120), CLAHE contrast boost (same registration, 2× slower),
domain-size pooling, and longer splat training (+0.4 dB held-out for 5.6× the time).

**Status and limits.**
- **Now the script default** (2026-10-12): `scripts/09_scan_object.py` uses `SfmReconstructor.robust()`, which is
  `SfmOptions.robust()` (threshold 0.002, ratio 0.9, sequential) and retries with
  `SfmOptions.robust_exhaustive()` when under 80 % of the keyframes register, keeping the better run. It warns if
  the result is still under 80 %. Plain `SfmOptions()` and `SfmReconstructor()` remain COLMAP's defaults, which the
  sweep uses as its baseline.
- Through the script, 1-angle clip: 120/120 registered in one model, no fallback needed (SfM 87 s).
  3-angle clip: sequential matching with the robust settings registered 70/120 in three models, the automatic
  exhaustive retry reached 120/120 in one model (SfM 447 s in total). Held-out PSNR 31.9 dB and 27.8 dB.
- Validated on two clips of one object. The fallback's `init_min_tri_angle=4` was only ever tested together with
  exhaustive matching, never alone.
- Exhaustive matching is O(N²): 376 s against 70 s for 120 frames. It runs only when sequential stalls.

**README:** added as a one-line status note (§3, §7). It is a real-footage failure mode that someone re-running
the scan stretch will hit, it is reproducible from the repo script, and it is now the default. Not a headline claim.

### D-002 · 2026-10-12 · Initial Gaussian scales from a KD-tree, not a dense distance matrix

**Component:** `hurobo.scan.SplatTrainer` (own trainer around `gsplat`). **Status:** Modified (bug fix in our code).

**Symptom.** Once SfM worked, training crashed with a CUDA out-of-memory error at about 50k sparse points: the
initialisation built a full N×N `torch.cdist` matrix (9.4 GiB at N = 50 252).

**Change.** The mean distance to the 3 nearest neighbours now comes from `scipy.spatial.cKDTree`. Memory is O(N)
instead of O(N²). The values are the same; a test compares them with a brute-force version.

**Status.** Done, tested (`tests/test_scan_sweep.py`). No README entry (an internal fix).

### D-003 · 2026-10-12 · Held-out PSNR and orbit coverage next to the training PSNR

**Component:** `SplatTrainer`, `hurobo.scan_sweep`. **Status:** Modified (extra evaluation stage).

**Symptom.** The trainer only reported PSNR on its training views. In a scratch run a 3-angle splat scored 24.2 dB
on the views it trained on and 17.4 dB on views withheld from training. A second problem showed up in the sweep: the broken 39/120 reconstruction had the **highest**
held-out PSNR of all configs (33.1 dB), because held-out views can only come from frames that registered, and those
sat in a ~180° fragment of the orbit.

**Change.** `SplatTrainer.train(..., holdout_every=N)` withholds every N-th keyframe and reports `heldout_psnr_db`.
`orbit_coverage_deg` measures how much of the orbit the registered cameras span. Both are reported together with the
registered fraction.

**Limits.** Held-out frames neighbour training frames on the same path, so this is interpolation along the lap, not
novel viewpoints, and PSNR is dominated by floor pixels. Coverage is reliable for full loops and approximate
(tens of degrees) for fragments. The 24.2 / 17.4 dB pair above came from a scratch run that is **not**
reproduced by repo code: the same 3-angle clip through `scripts/09_scan_object.py` gives 30.3 dB on training views
and 27.8 dB held-out. The scratch run differed in how keyframes were picked (sharpness on a quarter-scale image),
domain-size pooling and the seed; the cause was not isolated. Do not cite the 17.4 dB figure.

**README:** not added. Methodological detail; the sweep notebook carries it.

### D-004 · 2026-10-12 · Object crop from the floor plane and the cluster standing on it

**Component:** `hurobo.scan_crop`, used by `scripts/09_scan_object.py`. **Status:** Novel (replaces
`CropCylinder.around_points` as the default; the old fit stays as the fallback).

**Symptom.** The single-object auto-crop fits a cylinder to the 95th-percentile radius of all SfM points. On a bowl
on a textured floor most points are floor, so the cylinder has radius 9.0 against a camera orbit of 3.7, centred
0.6 units off the object, and keeps 42 % of the splat. The result is a wide floor disc with a tiny bowl.

**Upstream / previous default.** `CropCylinder.around_points`: assumes the sparse points sit on the object, which
holds for a turntable on a plain backdrop.

**Change.**
1. `estimate_object_crop`: fit the floor plane by RANSAC (inlier distance 1 % of the camera-ring radius, **not** a
   fraction of the point cloud's extent), take the points above it inside 0.6 of the ring radius, keep the largest
   connected cluster, fit the cylinder to it (radius = 1.3 × its 90th-percentile spread).
2. `choose_object_crop` falls back to the old fit when there is no dominant plane or no cluster, and an explicit
   `--crop X Y R` always wins.
3. The script writes the uncropped `<name>_full.ply` and crops from it, so `--recrop` changes the crop without
   re-running SfM or training.

**Evidence** (`examples/06_scan_parameter_sweep.ipynb`, `low_sift_threshold` splat). The reference is the bowl
located by triangulating its white pixels over 68 of 120 keyframes, independent of the splat and the SfM points
(ray residual 0.19 units; it works only because the bowl is the one white object):

| Method | Radius / bowl radius | Centre off (bowl radii) | Bowl inside | Gaussians kept |
|---|---|---|---|---|
| current percentile fit | 7.4 | 0.54 | yes | 42.5 % |
| floor + cluster (RANSAC) | **2.2** | **0.20** | yes | 37.7 % |
| explicit, typed in after looking | 2.1 | 0.04 | yes | 37.5 % |

The same pattern held on the `low_threshold_exhaustive` reconstruction (7.4 → 2.2 radii, 0.5 → 0.17 off).
Through the script both clips chose "floor + cluster": 1-angle centre (0.01, 0.51) radius 2.74, 3-angle centre
(0.41, 0.23) radius 2.17. Against the same white-pixel reference, on the script's own reconstructions:

| Clip | Method | Radius / bowl radius | Centre off (bowl radii) |
|---|---|---|---|
| 1-angle | current percentile fit | 7.2 | 0.55 |
| 1-angle | floor + cluster | 2.2 | 0.21 |
| 3-angle | current percentile fit | 18.0 | 0.53 |
| 3-angle | floor + cluster | 2.4 | 0.09 |

On the 3-angle clip the floor tilt against the orbit-plane frame was 2.6° (1.5° for the single lap) and the old
fit kept 66 % of the Gaussians against 13 % for the new crop.

**Not what an earlier version of this log said.** An earlier analysis concluded the bowl has almost no SfM
points and that none of three automatic methods validated. Both claims came from my RANSAC threshold being 0.89
units (1 % of an extent inflated to 89 by far points) and from using the median of white Gaussians as ground
truth, which moved by about 1 unit between the 7k- and 30k-iteration splats of the same scene. With a sane
threshold about 700 to 1200 points sit above the floor inside the ring, in one dominant cluster.

**Status and limits.**
- Validated on two clips of one object (a white bowl on a wooden floor) in one room. The reference method itself
  needs a white object; it is a validation tool, not part of the pipeline.
- Assumes a dominant floor or table plane and one object near the middle of the orbit. A turntable on a plain
  backdrop, or several objects, falls back to the old fit or keeps only the largest cluster.
- The crop radius is still about 2.2 object radii, so some floor stays in. Use `--recrop --crop` to tighten it.
- The cylinder is z-aligned in the orbit-plane frame; the floor's tilt against that frame was 1.5° here. It is
  not yet used to define "up".

**README:** not added. It is an internal improvement to a stretch-goal tool, validated on one scene.

### D-005 · 2026-10-12 · Keyframe sampling streams the video

**Component:** `hurobo.scan.KeyframeSampler`. **Status:** Modified (our code, same output).

**Symptom.** `sample` decoded every frame into a Python list first: about 12 GB for a 64 s 1080p clip.

**Change.** The video is streamed and only the sharpest candidate of the current time bin is held, with the same
bin sizes as before (`np.array_split`). Output is unchanged; a test checks the chosen frames.

**Status.** Done, tested. Frame spacing within a bin is still uneven because the sharpest frame is picked.

### D-006 · 2026-10-12 · Object-only splat for attaching to the simulator

**Component:** `hurobo.scan_crop.extract_object_splat`, `scripts/09_scan_object.py` (`<name>_object.ply`),
`hurobo.splat.render_bowl_composites`. **Status:** Novel.

**Symptom.** The cropped splat still contains the floor under and around the bowl, and its frame is the
orbit-plane frame (tilted 1.5 to 2.6 degrees against the floor, with the origin on the floor patch's centre). It
cannot be attached to a LIBERO body as "the bowl".

**Change.** Inside the crop cylinder: drop Gaussians within 3 sigma of the floor level (sigma measured on the
Gaussians, iterated outward from the SfM floor noise so it does not depend on the cylinder's size), keep the
largest connected blob, rotate the floor normal onto +z and put the floor at z = 0 under the blob's centre. Scale
stays in SfM units, so it attaches with `SplatFit.MATCH_BODY_XY`.

**Two settings were wrong at first and are worth recording.**
- Opacity floor 0.2 deleted most of the bowl: a smooth white object is drawn by many translucent Gaussians
  (median opacity 0.05), and the 3-angle bowl came out as a hollow rim arc. Default now 0.05.
- Blob link distance 4x the median spacing split the translucent surface into pieces. Default now 8x.
- With the earlier strict settings (opacity 0.2, link 4x) the `current` percentile cylinder picked a floater
  blob (214 Gaussians) instead of the bowl on the script's 1-angle splat. With the final defaults all three crop
  methods give a bowl on the sweep splat, so the extraction is much less sensitive to the crop than the crop is.

**Evidence** (`examples/06_scan_parameter_sweep.ipynb`): extracted footprint radius 1.21 and 1.00 against the
white-pixel reference radii 1.25 and 0.92 (1-angle, 3-angle clips); about 10k Gaussians (560 KB); all five splats
render as a bowl on the LIBERO table in place of its bowl.

**Status and limits.**
- Looks right in renders and in the LIBERO composite; no quantitative score against a ground-truth object mesh.
- The lower outside wall is brown: floor colour is baked into the bowl's lowest Gaussians. Interior is smooth
  and slightly cloudy.
- Needs a fitted floor plane; without one the script prints that it skipped the object-only splat.
- No collision geometry: `scripts/10_build_collider.py` can make one from this splat.

**README:** not added. Same scene, same two clips as D-004; it is the output a user wants rather than a new claim.

### D-007 · 2026-10-13 · Floor plane fitted inside the camera ring; combined clips and a flat plate

**Component:** `hurobo.scan_crop.estimate_object_crop`, `hurobo.scan.KeyframeSampler.sample_many`. **Status:** Modified (floor fit); the multi-clip scan is Novel and only partly validated.

**Symptom.** On the new `plate` clip the floor RANSAC found no plane: only 15 % of the SfM points lie on the floor (cabinets, bag and walls make up the rest), under the 30 % bar, so `choose_object_crop` fell back to the percentile fit.
**Cause.** The fit used every point, including the room outside the camera orbit. Restricted to points inside the orbit circle the floor share is 42 % (plate), 56 % (bowl_low), 39 % (bowl_high).
**Upstream defaults.** D-004: RANSAC on all points, `min_inlier_fraction=0.3`.
**Change.** `estimate_object_crop` fits the floor to points whose xy lies inside the fitted camera ring. Test: `test_estimate_object_crop_fits_the_floor_inside_the_camera_ring_only`.
**Evidence.** `examples/06_scan_parameter_sweep.ipynb`, part 2. The crop for `bowl_1angle` is nearly unchanged (radius 2.68 vs the 2.74 in D-004).
**Status and limits.** The pipeline default. Checked on `bowl_1angle` and the five new clips (3 bowl heights + plate), not on other scenes.

Two related findings from the same clips, recorded so they are not re-tried:

- **A flat object is not extractable automatically.** The plate stands about as high as the Gaussians' floor scatter, so the 3-sigma cut in `extract_object_splat` removes most of it (output: wood-coloured haze). A hand-picked crop at the orbit look-at point (radius 1.2) with a 1-sigma cut gives a recognisable but ragged disc. Not fixed; scanning it raised on a block or on a coloured cloth is the untested capture fix.
- **Combining clips only helps when the clips agree.** `bowl_high + bowl_low` in one COLMAP model gives the best bowl (mean white-pixel ray distance 0.11). Every combination containing `bowl_med` registers 120/120 (or 180/180) but has ray distance 0.6 to 1.1 and its object extraction is garbage, while `bowl_med` alone is fine (0.20). Cause not established (a moved bowl between recordings and a wrong cross-clip link are both possible). `registered 120/120` is necessary, not sufficient; check `white_object_residuals`.
- **Rejected: multi-view "white pixel" voting to cut the plate out** (scratch experiment, not in the repo). Keep a Gaussian if it projects onto white pixels (HSV S < 0.25, V > 0.55) in at least a fraction of the keyframes that see it, instead of cutting by height above the floor. On the plate it removes the floor and leaves a white rim, but the middle of the plate is missing and the rim is speckled with floor-coloured Gaussians; on `bowl_low` it keeps only the thin rim, because the bowl's outer wall is warm-toned in the photos and fails the white test. Fractions 0.5, 0.7 and 0.85 were tried. The plate's top face is seen almost edge-on in a low-height clip, so the Gaussians there are poorly constrained, which no post-cut fixes.
- **Update 2026-10-13: re-filming fixed the plate; combining did not.** The plate was re-filmed from a low, medium and high camera height. `plate_med` alone gives the best plate (robust SIFT, 120/120, 29.1 dB held-out; sequential and exhaustive matching look alike), and `plate_low` alone is usable. The automatic crop and the 3-sigma cut worked, no hand-picked crop. Combinations did not help: `low+high` registered 71/120, `low+med` registered 120/120 at 14.5 dB held-out with no floor plane found. So the earlier failure was the capture (camera too close to the floor), not the extraction. Still speckled with a dark rim shadow; checked on this one plate.
**README:** not added: the flat-object and combined-clip findings are one dataset, and the cause of the medium-clip failure is unknown.

---

## Open issues (known gaps, not decisions)

| Issue | Where | Effect |
|---|---|---|
| `object_frame_from_orbit` fits one plane to the camera centres; it is unreliable for multi-elevation paths. The fitted floor normal (D-004) would give "up" from the scene instead, but is not used to define the frame yet. | `hurobo.scan` | Wrong "up" and a tilted crop for a 3-angle style capture. The combined bowl scans (D-007) show 10 degrees of floor tilt against the frame; the object extraction removes it, but the crop is made in the tilted frame. |
| The crop radius is about 2.2 object radii, not tight. | `estimate_object_crop` | Some floor stays in the cropped splat. |
| The crop and the SfM defaults were validated on one object in one scene. | D-001, D-004 | Unknown on other objects, lighting, or a table with several objects (only the largest cluster is kept). |
| `--charuco` has run end to end on two phone clips of one table (D-009), not on a lap around the whole table. The guide's "one lap, 30 to 40 s" advice is still untested for the table. | `docs/scanning_guide.md` | The tabletop splat covers only the part of the table the cameras saw. |
| The sharpest frame per time bin is picked, so keyframe spacing is uneven. | `hurobo.scan.KeyframeSampler` | Uneven overlap. |
| Clip combinations containing the medium-height bowl clip, and both plate combinations, do not reconstruct consistently (D-007). Cause unknown. | `ScanSweep`, `combined_clip_sweep` | Multi-clip scanning is validated for bowl high + low only; use single clips otherwise. |
| Held-out views neighbour training views along the same lap, and PSNR is dominated by floor pixels. | `SplatTrainer`, `scan_sweep` | PSNR does not measure novel viewpoints or bowl quality. |
| Marker size: `test_03` fails the position gates (grasp 0.45 m from the bowl marker, release 0.33 m from the plate marker) with the assumed 6.0 cm printed markers. Both marker poses lie 0.26 m below the table and need the same distance scale, 0.65, as the hand's anchor scale (0.65), which implies markers of about 3.9 cm. With `--marker-size-m 0.039` `test_03` passes every gate (grasp 6.5 cm from the bowl centre, release 3.3 cm from the plate marker). **Not applied as a default: the printed size has not been measured.** | `RecordingSpec.marker_length_m`, `02_extract_demos.py --marker-size-m`, `17_check_capture.py` | Poses of the pad and plate markers are wrong by the size ratio until the real size is entered. |
| The cleaner's grasp/release (merged flickers, `CleaningConfig.min_open_s`) and the extractor's (smoothed hysteresis) are computed separately. Seen on `test_04`: the cleaner keeps 5.73-14.56 s as a full cycle, the extraction of that cut reports "no release detected". | `ClipCleaner.closed_signal`, `DemoExtractor` | A clip the cleaner accepts can still fail the extraction gates. |
| Of the 7 usable real clips, 3 (`233607`, `234347`, `234603`) never give a successful rollout as the approach source (0 of 11-62 attempts in every run); the cause is not diagnosed. `DemoGenerator` draws them uniformly, so they cost about 3/7 of its rollouts. | `ObjectCentricRetargeter`, D-013 | Success rate of `scripts/04` is capped by the share of good sources. |

### D-008 · 2026-10-13 · Training knobs do not fill the plate (negative result)

**Component:** `SplatTrainer` / gsplat `DefaultStrategy`. **Status:** Configured (new options, defaults unchanged).

**Symptom.** The extracted plate splat (`plate_med`, `plate_low`) has translucent patches in the interior and a distorted, squarish rim; dark loop artefacts on the surface.
**Cause.** Not the floor cut: at 3 sigma the cut is 0.024 SfM units against a plate height of about 0.27, and re-cutting at 1 sigma changes 33,070 to 33,377 Gaussians. The gaps are already in the trained splat. Suspected: white, glossy, textureless surface, so too few SfM points and weak photometric gradient.
**Upstream defaults.** `grow_grad2d` 2e-4, `reset_every` 3000, 7000 iterations.
**Change.** `SplatTrainer(grow_grad2d, reset_every)`, `ScanConfig` fields of the same names, `plate_fill_sweep()`.
**Evidence.** `plate_med`, held-out PSNR / Gaussians: baseline 29.1 / 1.2 M; 30k iterations 29.9 / 3.2 M; `grow_grad2d` 1e-4 29.5 / 2.8 M. Whole-frame PSNR hardly moves and the interior gaps remain in renders of the extracted object. `no_reset`, `fill_all` and the `plate_low`/`plate_high` runs were stopped, not completed. 30k run: the automatic floor cut kept the floor patch (object height 0.72 against 0.27), i.e. more Gaussians made extraction worse.
**Status and limits.** Only `plate_med`, visual judgement, no object-level metric. Options stay at upstream defaults.
**README:** not added, negative scratch result.

---

### D-009 · 2026-10-14 · Metric table scan from a ChArUco board on a tablet

**Component:** `scripts/09_scan_object.py --charuco`, `hurobo.scan` (`CropBox`, `Similarity.shifted`, `board_visibility`), `hurobo.scan_crop.locate_objects_on_table`. **Status:** Novel (own code around OpenCV ArUco).

**Symptom.** The first real `--charuco` run, on `tabletop_charuco_topleft.mp4`, registered 120/120 keyframes and then failed with "only 0 ChArUco corners seen". Board detection found no marker in any of 619 frames at full resolution, in four of the six table clips (`tabletop_charuco_*`, `table_scene_zoom_01`). Separately, the pipeline had no way to take a board of another size, to place the tabletop (not the board surface) at z = 0, to crop a 1.445 x 0.60 m rectangle, or to cut objects whose position was never measured.
**Cause.** Detection: in the unusable clips the board is about 70 px wide at a grazing angle and the tablet shows the white squares cyan (checked on a crop of one frame); in `table_scene_zoom_02/_03` it fills a few hundred pixels and all 17 markers are found. Pipeline: board sizes were hard-wired to 35 mm squares, and objects needed a hand-measured `--object X Y R H`.
**Upstream defaults.** `ChArUcoCalibrator` 35 mm / 26 mm; cylinder tabletop crop; objects by hand-measured position.
**Change.** `--square-length`, `--marker-length`, `--board-thickness` (lifts the table frame so the tabletop is z = 0), `--table-box` (`CropBox`), `--find-object NAME DIAMETER HEIGHT` (blobs above the tabletop inside the box, matched by size), several videos per run (`KeyframeSampler.sample_many`), and `--recrop --charuco` to redo the alignment and cut without re-training. `board_visibility` tests a clip for the board before SfM.
**Evidence.** `examples/06_scan_parameter_sweep.ipynb`, part 4; tests `test_scan_split.py`, `test_locate_objects.py`, `test_board_visibility.py`. `zoom_02` alone: 120/120, corner RMS 1.1 mm, held-out PSNR 29.5 dB. `zoom_02 + zoom_03`: 120/120, RMS 0.5 mm, held-out 25.6 dB (not like-for-like). Plate and bowl blobs measure about 26 cm and 16 x 14 cm against a ruler's 25.5 and 13.8 cm. First finder version searched the whole splat and returned the plate at x = 1.73 m (outside the table); restricting it to the table box fixed that (plate (0.192, -0.303), bowl (0.466, -0.301), the same as in the single-clip scan).
**Status and limits.** One table, one board position, two clips; the table is only partly covered. The tablet thickness and square size are hand measurements, the 18.6 mm marker size is derived from the 26/35 ratio. The cut cylinders start 1 cm below the tabletop, so the extracted plate and bowl carry a skirt of wood grain (visible in the renders); they are not floor-cleaned like the single-object extraction (D-006). Not the guide's default command yet.
**README:** added: the splat gallery in section 5 (sweep, held-out photo vs splat) as a visual only, with the one-scene caveat. The table scan is not used in the sim.

### D-009 · 2026-10-14 · Shape-prior fill for the plate

**Component:** `hurobo.splat_fill` (`DishFiller`, `DishPrior`). **Status:** Novel.

**Symptom.** After D-008 the extracted plate still has translucent interior patches and a warped rim: solid top-down coverage 0.34 (`plate_med`) and 0.47 (`plate_low`) of the footprint disc.
**Cause.** See D-008: a glossy white, textureless plate gives the optimiser too little signal; more iterations or densification did not help.
**Upstream defaults.** None: nothing upstream does this. Learned alternatives were reviewed, not tried (see below).
**Change.** `DishFiller.fill` fits the measured dish (0.255 m diameter, 0.028 m high from `measurements.yaml`; flat-base fraction 0.55 and thickness 6 mm are assumed) to an extracted object splat: scale from the median over 36 azimuth sectors of the 95th-percentile radius, drops Gaussians more than 8 mm off the surface, adds flat surface-aligned Gaussians where no solid Gaussian is within one lattice spacing, coloured from bright low-saturation neighbours only.
**Evidence.** `examples/06_scan_parameter_sweep.ipynb`, Part 5; `tests/test_splat_fill.py`. `plate_med`: 22,445 kept, 10,625 dropped, 44,313 added, coverage 0.34 to 1.00. `plate_low`: 24,486 kept, 34,891 added, 0.47 to 0.98. The coverage proxy ignores Gaussian extent and rises by construction, so the judgement is visual: closed white disc, printed pattern kept, facets and baked gloss visible.
**Status and limits.** Opt-in, not wired into `scripts/09_scan_object.py` or `ScanSweep`. Two clips of one plate, no held-out photometric score. The rim and profile are the prior, not a measurement. Splat units, as before (`MATCH_BODY_XY`).
**Alternatives reviewed, not run (subagent, 2026-10-14; sizes from the Hugging Face API, VRAM and runtimes mostly unverified):** Depth Anything V2 Large 1.34 GB (CC-BY-NC, pip, fits 16 GB) as a depth loss in `SplatTrainer`; Depth Pro 1.9 GB; PGSR / 2DGS planar regularisation (compile risk, port as a loss); TRELLIS 3.3 GB and SAM 3D Objects 13.2 GB are risky on 16 GB and cu126; PoinTr has a Python 3.7-era extension stack and the wrong domain. Rejected for now, not evaluated on our data.
**README:** added: the scanned plate in the gallery, labelled as filled by a shape prior. Not added: any quality number (two clips, visual judgement only, opt-in).


### D-010 · 2026-10-14 · E1 Scaled-Native: real-size bowl and plate in LIBERO, retargeted from the home pose

**Component:** `hurobo.object_scale`, `scaled_native`, `retarget`, `generate`, `reach`, `capture_plan`, `recording`, `markers.BoardMarkerPoseEstimator`, `extract` (plane anchor). **Status:** Modified (LIBERO scene) and Novel (retargeting, board-frame recording). Validated on **made-up** hand paths only; no real clip has been processed.

Full guide: [experiment_scaled_native.md](experiment_scaled_native.md). Decisions, in the order they were forced on us:

1. **Uniform scale per object, to the measured diameter.** Bowl x1.227 (LIBERO 11.2 cm to 13.8 cm), plate x1.855 (13.8 cm to 25.5 cm). Heights follow LIBERO's proportions: bowl -4 %, plate +10 % against the measurements (the bowl height in `measurements.yaml` is a `TODO`). Anisotropic scaling was not done: the collision boxes sit at arbitrary rotations in the body frame.
2. **The other five libero_goal objects are parked** (8 m away, collisions off). With the plate at real size, 0.00 % of random bowl/plate pairs in the work area avoid them, 38 % when the table is cleared (`layout_feasible_fraction`, notebook 08, section 2; 0.06 % against 29 % in an earlier, wider work area). The scene is therefore not fully "native". `keep_distractors=True` restores them.
3. **Layouts are ours, not LIBERO's init states.** Its 2 cm ranges put the small plate where the large one now overlaps the bowl.
4. **The work area is 0.26 m deep and 0.50 m wide**, not the whole table. A reach map (`hurobo.reach`, home orientation, fingertips 0.94 and 1.05 m up) gives position errors under 0.5 cm for |y| <= 0.2 up to x = 0.10, 2.3 cm at (x = 0.10, |y| = 0.3) and 4.5 to 15 cm farther out at the higher height. A first, wider work area made an approach from the far side time out (arm stuck at 0.76 m from the base). The approach is also pulled back inside 0.76 m of the base.
5. **Grasp height is fixed to 4 cm above the table** in the robot demo; the human height is replaced. With the human value (clipped to 3 to 6 cm) 2 of 10 rollouts closed on the very top of the 2 mm wall and did not lift the bowl; with 4 cm none of the next 30 did.
6. **The carry path is for the bowl, not the fingers.** Held by one rim point the bowl swings like a pendulum, 3 to 4 cm off the fingers' position. Driving the fingers along the human path left the bowl 3.3 cm from the plate centre, 3 mm outside LIBERO's 3 cm check. The executor measures the offset every step and moves the fingers so that the bowl follows the bent path.
7. **The depth anchor can use a plane.** The pad marker lies on the table beside the bowl, so it says nothing about the depth of a rim grasp. `ExtractionConfig.grasp_height_m` intersects the camera ray through the grasp with the plane at that height (bowl height minus 1.2 cm). Test: `test_height_anchor_recovers_depth_without_a_marker_at_the_grasp`. Default (marker anchor) unchanged.
8. **Success terminates the episode and LIBERO resets itself** to the stock scene. A step after a success silently runs on the stock, small objects. The executor stops at the first success; the evaluation loop must too. For the same reason `lerobot-eval` cannot be used on this scene as it is.

**Evidence** (repo code; `scripts/04_generate_dataset.py` on four made-up demos with distinct hand starts, new random layouts; steps are control steps at 20 Hz):

| Run | Config | Rollouts | Succeeded | Failure modes |
|---|---|---|---|---|
| scratch, seed 3 | grasp height snapped, loose tolerances | 30 | 25 | 3 timeouts, 2 placed 3.5 cm off |
| scratch, seed 7 | + pass tolerance 2 cm, release at 4 mm, wait after release, x clamp | 30 | 29 | 1 approach timeout at the reach limit |
| scratch, seed 11 | + radial reach clamp, narrower work area | 40 | 40 | none |
| repo script, seed 5 | final config, four made-up demos | 40 | 39 | 1 timeout |

The first three came from a scratch loop that is not in the repo (it called the same classes); only the last row is reproducible from `scripts/04_generate_dataset.py`. The two final-config runs (40/40 and 39/40, different seeds and demo sets) put the success rate at about 98 %; the one timeout was not diagnosed. Each fix was made after looking at the failures of the previous run, and the final config was only run on seeds it was not tuned on. Typical episode: 240 to 300 steps (12 to 15 s). One of three generated episodes in an early run took 710 steps.

**Status and limits.**
- The pipeline defaults are the config above. The notebook runs 8 rollouts and 4 twin replays as a smoke test.
- The "human" is a minimum-jerk path with 2 mm noise: no occlusion, no depth error, no hand-size scatter. The success rate is that of retargeting plus execution given a clean demo, an upper bound for real clips.
- The plate is 10 % taller than the real one and 1.86x wider; its collision boxes extend about 1 cm past the visual rim (LIBERO's own approximation).
- Reach errors were measured with one orientation (home), at two heights, from a single run.
- Real-clip behaviour of `BoardMarkerPoseEstimator` is tested only on rendered boards (pose error < 3 mm, < 1 degree).
- The phone-pose search uses 1080p intrinsics and a geometric visibility model; nothing was filmed from the poses it proposes.

**README:** added to the status table and the notebook list as "built and tested on stand-in data". Not added: any success rate, because it is an upper bound from made-up input.

### D-011 · 2026-10-16 · Pretrained SmolVLA LIBERO checkpoints in the cleared and scaled scenes

**Component:** `hurobo.policy_rollout`, `benchmark`, `training`, scripts 18 and 19. **Status:** As-is weights, our own evaluation loop; training wrapper Configured.

**Symptom.** Both most-liked SmolVLA LIBERO checkpoints solve the stock task but 0 of 10 episodes in the cleared scene, with and without real-size objects.

| Checkpoint | stock | clean (LIBERO sizes) | scaled (real sizes) |
|---|---|---|---|
| `lerobot/smolvla_libero` | 10/10 (Wilson 0.72-1.00) | 0/10 (0.00-0.28) | 0/10 (0.00-0.28) |
| `HuggingFaceVLA/smolvla_libero` | 8/10 (0.49-0.94) | 0/10 | 0/10 |

Replan every 10 actions, 300 steps, 10 episodes per cell, same layout seeds for both checkpoints. Two later conditions keep LIBERO's own bowl/plate placement and change only the clutter and size (notebook 09, section 2b): `clean_stock_layout` 6/10 (`lerobot`) and 3/10 (`HuggingFaceVLA`), `scaled_stock_layout` 0/10 and 0/8. So a familiar layout restores part of the success; real-size objects do not. `lerobot/smolvla_libero` with replan 50, `scaled`: 0/5. The bowl was lifted in 2/10 `clean` and `scaled` episodes (lerobot) and 0/10 (HuggingFaceVLA); the end effector stays about 15 cm from the bowl.
**Cause.** Not established. `clean` already fails at LIBERO's own object sizes, so the real-size scaling is not the (only) cause. Untested candidates: our layouts lie outside the fixed start positions the checkpoints saw, and the removed distractors.
**Upstream defaults.** `lerobot-eval` on the stock scene.
**Change.** Own loop that applies the same processing as `lerobot-eval` (checked indirectly: stock success matches expectation), stops at first success, and resets the scene itself. `lerobot/smolvla_libero` needs `image -> camera1`, `image2 -> camera2`.
**Evidence.** `scripts/18_benchmark_policy.py --checkpoints lerobot hfvla --conditions stock clean scaled --replan 10 --episodes 10`; notebook 09; saved results and one video per experiment in `eval/pretrained_libero_checkpoints/`. Checkpoints chosen by Hub likes (20 and 7), queried 2026-10-16. Open loop (100 frames of the stand-in dataset): gripper sign accuracy 0.56 and 0.44, and position error above the do-nothing baseline on dx, dy, dz; the demos are retargeted stand-in paths, not LIBERO human demos.
**Status and limits.** `smolvla_base` cannot run LIBERO zero-shot (6-D state/action). Training: only a 20-step smoke run on 30 stand-in episodes (checkpoint 0/1 in `scaled`, as expected). Those episodes run 270-350 steps against LIBERO's 300-step limit. The `stock` result for HuggingFaceVLA depends on its n_action_steps being overridden to 10.
**README:** added to section 4 as a chart with the 10-episode caveat and "cause not established". It is a measurement of the public checkpoints, not of our method.

### D-012 · 2026-10-16 · Cleaning step before extraction: one hand, one grasp cycle, a cut per clip

**Component:** `hurobo.clip_clean`, `scripts/20_clean_clips.py`, `scripts/21_clip_cutter.py`. **Status:** Novel (extraction takes the cut through `window_s`).

**Symptom.** On the four test clips of 2026-10-16 the hand is in view for only 22-61 % of the frames, and the extraction gates fail on the whole clip: "hand detected in 83 % of frames; hand lost for 2.70 s" (`test_03`), and the clip is skipped when the plate marker is missing in its first 0.6 s.
**Cause.** The extraction uses every frame, so dropouts before the hand enters and after it leaves count against the clip, and a clip with several grasps gives an ambiguous grasp/release pair.
**Upstream defaults.** None (our own pipeline): all frames of the clip, first grasp and first release of the smoothed aperture.
**Change.** A cleaning step: per-frame filters (no hand, more than one hand, PnP reprojection > 15 px, landmark within 8 px of the border, depth outside 0.15-3 m), then the first gap-free run (gaps up to 0.4 s bridged) that holds an open, closed, open cycle; kept from the start of the run to 2 s after the release (0.5 s at first; raised on 2026-10-16 so the hand's retreat is kept). Closed pinches shorter than 0.2 s and open gaps shorter than 0.3 s are noise: on `test_03` one sample at 5.2 cm (8.5 s) crossed the 5 cm open threshold in the middle of a 5 s grasp, which without the 0.3 s rule cut the clip at the false release. Clips shorter than 1.5 s, or without the board, bowl or plate marker in the first second, are rejected. The cuts go to `<folder>/cuts.json`; `02_extract_demos.py --cuts` skips rejected clips and passes the window to the extractor, which still takes the table pose and the layout from the clip's first 0.6 s.
**Evidence.** `scripts/20_clean_clips.py "data/raw/e1/test_0*.mp4"`: `test_03` is cut to 2.70-11.10 s and `test_04` to 5.73-14.56 s; `test_01` and `test_02` are rejected (no complete cycle with one hand). Extracting `test_03` with and without the cut: the two dropout reasons disappear (169 against 310 samples). 10 unit tests in `tests/test_clip_clean.py`. Notebook 10.
**Status and limits.** Default thresholds are guesses checked on four clips, not tuned: no ground truth for where a human would cut. The extraction still decides grasp and release by its own smoothed hysteresis, which can differ from the cleaner's on noisy clips. `test_03` still fails the extraction gates on position ("grasp 0.45 m from bowl marker", "release 0.33 m from plate marker"): the cut does not address it (see the open issues).
**README:** not added: a data-preparation step, validated on four test clips only.

---

### D-013 · 2026-10-16 · SmolVLA fine-tuning settings for a 50-minute budget, and per-checkpoint evaluation

**Component:** `lerobot-train` driven by `hurobo.training.SmolVLATrainConfig`; `hurobo.checkpoint_eval`, `scripts/23_eval_checkpoints.py`. **Status:** Configured (training); own code (evaluation).

**Symptom.** None: this is a first fine-tune, not a fix. SmolVLA's defaults assume a long run (`scheduler_warmup_steps` 1000, cosine decay over 30000 steps, batch 32 in `lerobot/smolvla_libero`). With 3000 steps those defaults would spend a third of the run warming up and the rate would never decay.
**Cause.** Compute and time: a 16 GB GPU shared with another process (about 3.5 GB), about 50 minutes available after the dataset was generated.
**Upstream defaults.** lr 1e-4 (peak), warm-up 1000, cosine decay to 2.5e-6 over 30000 steps, batch 32, `train_expert_only` true, vision encoder frozen.
**Change.** Kept the peak learning rate (1e-4) and expert-only training. Changed: batch 16, 3000 steps (1.35 epochs of the 35.5k frames), warm-up 100, decay length 3000 so the rate falls to 2.5e-6 at the end, checkpoint every 500 steps (six checkpoints), loss logged every 50 steps. Batch 16 and 3000 steps follow from a 60-step smoke run (1.5 steps/s at batch 16 on the shared GPU, 4.5 GB of memory for training), not from a search: nothing was tuned, and a larger batch or more steps were not tried. The fields `learning_rate`, `warmup_steps`, `decay_steps`, `log_freq` were added to `SmolVLATrainConfig` (only passed to `lerobot-train` when set). Evaluation: every checkpoint gets the same 10 layouts (seeds 0-9) in the `scaled` condition at replan 10, with an episode limit of 500 steps because the generated demos are longer than LIBERO's 300-step limit (median 321, 90th percentile 529, maximum 663 frames); episodes needing more than 500 steps count as failures. Open-loop error on 100 frames sampled from the **training** dataset (no held-out episodes exist), so it measures fit.
**Evidence.** `uv run python scripts/19_train_smolvla.py data/e1/generated_clips --out outputs/e1_smolvla_ckpts/run --steps 3000 --batch-size 16 --save-freq 500 --lr 1e-4 --warmup-steps 100 --decay-steps 3000 --log-freq 50`, then `MUJOCO_GL=egl uv run python scripts/23_eval_checkpoints.py outputs/e1_smolvla_ckpts/run --dataset data/e1/generated_clips --out results/smolvla_ckpts --episodes 10 --max-steps 500 --watch --final-step 3000`; results table and figures in `examples/13_smolvla_checkpoints.ipynb`.
**Status and limits.** Validated only on this dataset: 100 generated episodes from 12 real clips, of which 5 clips had their release-distance QC gate waived by the user. One training seed, 10 closed-loop episodes per checkpoint (Wilson intervals are wide; differences between checkpoints are noise unless the intervals do not overlap), no held-out episodes for the open-loop error. **Result (scratch, not README):** closed-loop success 0, 2, 1, 1, 4, 2 of 10 at steps 500 to 3000 (Wilson intervals overlap everywhere: no step-to-step difference is established, only that the policy sometimes completes the task); open-loop position error on training frames fell from 0.113 to 0.084 (do-nothing baseline 0.137) and gripper sign accuracy rose from 0.92 to 0.96; training loss 0.42 at step 500, 0.24 at step 3000, still slowly falling when the rate reached its floor.
**README:** not added: single seed, 10 episodes per checkpoint, nothing tuned.

### D-013 · 2026-10-17 · MimicGen-style expansion: source screening and a length filter matter more than nearest-neighbour choice or noise

**Component:** `hurobo.mimicgen`, `hurobo.utils.mimicgen_view`, `scripts/22_mimicgen_expand.py`, notebook 12. **Status:** Modified (MimicGen's ideas, our code; `generate.py`, `retarget.py` and `scripts/04` are untouched and remain the defaults).

**Symptom.** On the 7 usable real clips (`data/e1/demos_clips`, `summary.json` "usable"), 120 random layouts, `scripts/04`-style rollouts (`StockBaseline` drives `DemoGenerator.rollout`) keep 47 (39 %), and only 18 of the 47 are within the evaluation's 300 steps (median 319, max 634). Three of the seven sources never succeed (0/19, 0/21, 0/20 as approach source), and the executor runs at `action_limit=0.5`.
**Cause.** Per-source quality differs from 0 to 100 % (approach source `234314`: 14/14 kept in the stock run, but 11/11 of its rollouts in run 02 were over 300 steps); the cause of the bad ones is not diagnosed (open issues). Episode length comes from the slow default speed. Nearest-neighbour choice *alone* made it worse (see the table): the nearest source to a layout is often one of the bad ones.
**Upstream defaults.** MimicGen: source chosen uniformly or among the `nn_k` nearest by object pose (`nn_k=3` for its Square and Nut Assembly tasks), per-subtask selection a toggle, action noise sigma 0.05 in simulation, success filter only, no screening of sources, linear interpolation between segments.
**Change.** New, separate module: `SourceSelector` (nearest layout, separately for the approach by bowl position and the carry by the bowl-to-plate vector), `MixedRetargeter` (approach of one demo, carry of another; valid because the parent already bends each carry to start at the bowl and end on the plate), `NoisyPlanExecutor` (noise on the six pose dimensions, not in the grasp and release phases, recorded as executed), a filter that drops successes longer than 300 steps, and `MimicGenExpander.screen` (each source rolled out alone on 6 shared layouts, dropped below 25 % success; this one is not in MimicGen). Speed is the existing `RetargetConfig.action_limit`, set to 0.8 on the command line.
**Evidence.** Same 100 layouts (`--seed 0`) in all rows, dry runs, `MUJOCO_GL=egl uv run python scripts/22_mimicgen_expand.py data/e1/demos_clips --out data/e1/mimicgen/<run> --attempts 100 ...`; reports in `data/e1/mimicgen/run_0*/report.json` (not committed). "Kept" counts rollouts accepted by LIBERO and, for MimicGen-style, within 300 steps.

| Run | Settings | Kept / attempts | Within 300 steps | Median steps |
|---|---|---|---|---|
| 01 stock | uniform source, speed 0.5 (120 layouts) | 47/120 | 18 | 319 |
| 06 stock | uniform source, speed 0.8 | 38/100 | 21 | 234 |
| 01 mimicgen | nn_k 3, noise 0.05, speed 0.5, 120 layouts | 15/120 | 15 | 285 |
| 03 | nn_k 1, noise 0.05, speed 0.5, 120 layouts | 5/120 | 5 | 291 |
| 06 mimicgen | nn_k 3, noise 0.05, speed 0.8, no screening | 38/100 | 38 | 230 |
| 04 | + screening, speed 0.5 | 51/100 | 51 | 256 |
| 05 | + screening, speed 0.8 | 72/100 | 72 | 206 |
| 07 | + screening, speed 0.8, noise 0 | 64/100 | 64 | 195 |
| 08 | + screening, speed 0.8, uniform source (nn_k 0) | 66/100 | 66 | 206 |
| 09 | as 05, other seed, writes a dataset (100 episodes of 130 attempts) | 100/125 | 100 | 206 |

Screening kept `233646`, `233859`, `234314`, `234827` in run 05, and `233646`, `234314`, `234827` in run 09 (`233859` had 1 of 6: a 6-layout screen is coarse). Throughput depends on the success rate and on machine load: 650 kept episodes/hour in a 12-rollout smoke test on stand-in demos (one process); 13-340/hour in the table's dry runs (three processes at once, load average 5-16); 163/hour for run 09 (one process, writes video, load 16).
**Status and limits.** A scratch result, not a repo default and not used by the training run. One seed and 100-120 layouts per row, so differences of 10 points (72 against 64 or 66) are within sampling noise (Wilson 95 % half-width about 9 points at n=100): the data support "screening and speed help a lot", not "noise" or "nn_k 3" helping or hurting by themselves. Whether the injected noise or the source mixing improves the *trained policy* is not measured (no training was run). Validated on 7 real clips of one session, one scene, one object pair. Not done: hand orientation (a `HumanDemo` has only the TCP position), SE(3) transform of whole segments, motion-planned links (SkillMimicGen), per-source weighting beyond the keep/drop screen. With 7 sources, `nn_k=3` is nearly half of the sources, so the nearest-neighbour rule has little to select among; the layouts of the 7 clips are also clustered (bowl at table x 0.50-0.61, plate 0.25-0.37 m away mainly along table x), so most new layouts are far from every source and "nearest" is still far.
**README:** not added: validated on 7 clips, one seed, and nothing trained on it yet.

---

---

## Entry template

```markdown
### D-NNN · YYYY-MM-DD · Short title

**Component:** package or module. **Status:** As-is | Configured | Modified | Novel | Rejected.

**Symptom.** What did not work on our data, with numbers.
**Cause.** What we found, and how we know.
**Upstream defaults.** What the original does.
**Change.** What we did differently, and where in the repo.
**Evidence.** Experiment, config, numbers, how to reproduce.
**Status and limits.** Is it the default yet? What is untested? Which clip or data?
**README:** added | not added, and why.
```
