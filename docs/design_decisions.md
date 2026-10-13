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
| Marker poses, ChArUco calibration | OpenCV ArUco | As-is | |
| Structure from motion | `pycolmap` (SIFT, matching, incremental mapping) | **Configured** | D-001. Lowered SIFT threshold is the script default; exhaustive matching is the fallback. |
| Splat training | `gsplat` rasteriser and densification strategy | **Modified** | Own minimal trainer (`SplatTrainer`), see D-002, D-003. |
| Object crop | none (RANSAC plane + connected components, from `scipy`/`numpy`) | **Novel** | D-004. |
| Metric table scan (ChArUco on a tablet, tabletop box crop, object finder) | OpenCV ArUco board detection; own alignment (`ChArUcoAligner`) | **Novel** | D-009. |
| Splat rendering / compositing | DISCOVERSE `gaussian-renderer` | As-is | `load_ply` and rasterisation; compositing code is ours. |
| Collision geometry | open3d, CoACD | As-is | |
| Retargeting | MimicGen idea (concept only) | Novel (own code) | MimicGen is cited, not installed. |

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

**Evidence** (`examples/08_scan_parameter_sweep.ipynb`, 1-angle clip, one run each, from repo code):

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

**Evidence** (`examples/08_scan_parameter_sweep.ipynb`, `low_sift_threshold` splat). The reference is the bowl
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

**Evidence** (`examples/08_scan_parameter_sweep.ipynb`): extracted footprint radius 1.21 and 1.00 against the
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
**Evidence.** `examples/08_scan_parameter_sweep.ipynb`, part 2. The crop for `bowl_1angle` is nearly unchanged (radius 2.68 vs the 2.74 in D-004).
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
**Evidence.** `examples/08_scan_parameter_sweep.ipynb`, part 4; tests `test_scan_split.py`, `test_locate_objects.py`, `test_board_visibility.py`. `zoom_02` alone: 120/120, corner RMS 1.1 mm, held-out PSNR 29.5 dB. `zoom_02 + zoom_03`: 120/120, RMS 0.5 mm, held-out 25.6 dB (not like-for-like). Plate and bowl blobs measure about 26 cm and 16 x 14 cm against a ruler's 25.5 and 13.8 cm. First finder version searched the whole splat and returned the plate at x = 1.73 m (outside the table); restricting it to the table box fixed that (plate (0.192, -0.303), bowl (0.466, -0.301), the same as in the single-clip scan).
**Status and limits.** One table, one board position, two clips; the table is only partly covered. The tablet thickness and square size are hand measurements, the 18.6 mm marker size is derived from the 26/35 ratio. The cut cylinders start 1 cm below the tabletop, so the extracted plate and bowl carry a skirt of wood grain (visible in the renders); they are not floor-cleaned like the single-object extraction (D-006). Not the guide's default command yet.
**README:** not added. One scene, flags rather than defaults, and the table scan is not yet used in the sim.

### D-009 · 2026-10-14 · Shape-prior fill for the plate

**Component:** `hurobo.splat_fill` (`DishFiller`, `DishPrior`). **Status:** Novel.

**Symptom.** After D-008 the extracted plate still has translucent interior patches and a warped rim: solid top-down coverage 0.34 (`plate_med`) and 0.47 (`plate_low`) of the footprint disc.
**Cause.** See D-008: a glossy white, textureless plate gives the optimiser too little signal; more iterations or densification did not help.
**Upstream defaults.** None: nothing upstream does this. Learned alternatives were reviewed, not tried (see below).
**Change.** `DishFiller.fill` fits the measured dish (0.255 m diameter, 0.028 m high from `measurements.yaml`; flat-base fraction 0.55 and thickness 6 mm are assumed) to an extracted object splat: scale from the median over 36 azimuth sectors of the 95th-percentile radius, drops Gaussians more than 8 mm off the surface, adds flat surface-aligned Gaussians where no solid Gaussian is within one lattice spacing, coloured from bright low-saturation neighbours only.
**Evidence.** `examples/08_scan_parameter_sweep.ipynb`, Part 5; `tests/test_splat_fill.py`. `plate_med`: 22,445 kept, 10,625 dropped, 44,313 added, coverage 0.34 to 1.00. `plate_low`: 24,486 kept, 34,891 added, 0.47 to 0.98. The coverage proxy ignores Gaussian extent and rises by construction, so the judgement is visual: closed white disc, printed pattern kept, facets and baked gloss visible.
**Status and limits.** Opt-in, not wired into `scripts/09_scan_object.py` or `ScanSweep`. Two clips of one plate, no held-out photometric score. The rim and profile are the prior, not a measurement. Splat units, as before (`MATCH_BODY_XY`).
**Alternatives reviewed, not run (subagent, 2026-10-14; sizes from the Hugging Face API, VRAM and runtimes mostly unverified):** Depth Anything V2 Large 1.34 GB (CC-BY-NC, pip, fits 16 GB) as a depth loss in `SplatTrainer`; Depth Pro 1.9 GB; PGSR / 2DGS planar regularisation (compile risk, port as a loss); TRELLIS 3.3 GB and SAM 3D Objects 13.2 GB are risky on 16 GB and cu126; PoinTr has a Python 3.7-era extension stack and the wrong domain. Rejected for now, not evaluated on our data.
**README:** not added: two clips, visual judgement only, opt-in.

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
