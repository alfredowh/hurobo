# Design decisions

What we reuse from existing work, with which settings, and where we changed it and why. Newest entries first. The
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
| Structure from motion | `pycolmap` (SIFT, matching, incremental mapping) | **Configured** | D-001. Our default is still COLMAP's. |
| Splat training | `gsplat` rasteriser and densification strategy | **Modified** | Own minimal trainer (`SplatTrainer`), see D-002, D-003. |
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

On the 3-angle clip (scratch ablation, not in repo code) sequential matching with the lower threshold stopped at 74
to 79 of 120, and exhaustive matching was needed for 120/120.

**Did not help** on this footage: full resolution alone (10/120), CLAHE contrast boost (same registration, 2× slower),
domain-size pooling, and longer splat training (+0.4 dB held-out for 5.6× the time).

**Status and limits.**
- The robust settings are **not the pipeline default yet**. Only the sweep uses them.
- Validated on one clip, one run per config. The 3-angle result comes from scratch scripts.
- Exhaustive matching is O(N²): 376 s against 70 s for 120 frames. Use it only if sequential stalls.

**README:** reviewed, recorded as a one-line limitation in the status table (§3). It is a real-footage failure
mode that someone re-running the scan stretch will hit, and it is reproducible. Not a headline claim.

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
(tens of degrees) for fragments.

**README:** not added. Methodological detail; the sweep notebook carries it.

---

## Open issues (known gaps, not decisions)

| Issue | Where | Effect |
|---|---|---|
| The auto-crop is fitted to the 95th-percentile radial distance of all SfM points. On a bowl on a textured floor that is far larger than the object: on the `low_sift_threshold` reconstruction the cylinder has radius 9.0 and is centred at the frame origin, while the camera orbit radius is 3.7 and the orbit centre is 0.6 away from the origin. It keeps 42 % of the Gaussians. Replacement under evaluation (not implemented): fit the floor plane with RANSAC (threshold a few percent of the orbit radius, **not** a fraction of the point-cloud extent, which far-away points inflate to 89 units), take the SfM points above the floor inside the camera ring, keep the largest connected cluster. On the two full-loop reconstructions that gives one dominant cluster (689 of 694 and 1170 of 1225 points) whose centre is within 0.15 units of the camera-circle centre. Caveat: there is no reliable ground truth for the bowl position. The proxy used (median of white Gaussians) moved by about 1 unit between the 7k- and 30k-iteration splats of the same scene, so agreement with it (0.3 to 1.2 units) is not evidence either way. An earlier version of this row said the bowl has almost no SfM points; that was an artefact of a too-loose RANSAC threshold (0.89 units) and is withdrawn. | `CropCylinder.around_points`, `scan_single_object` | `data/splats/<name>.ply` from the script is not cut to the bowl. |
| `KeyframeSampler` loads every frame into memory (about 12 GB for a 64 s 1080p clip) and picks the sharpest frame per time bin, so spacing is uneven. | `hurobo.scan.KeyframeSampler` | Memory, and uneven overlap. |
| `object_frame_from_orbit` fits one plane to the camera centres; it is unreliable for multi-elevation paths. | `hurobo.scan` | Wrong "up" for the 3-angle style capture. |
| `docs/scanning_guide.md` recommends 2 to 3 minutes at 120 frames (one keyframe per 1 to 1.5 s) and three laps in one clip. Both did worse than one dense lap in our tests. | `docs/scanning_guide.md` | Guidance contradicts the evidence. |
| The script does not warn when SfM registers a small fraction of the frames. | `scripts/09_scan_object.py` | A failed scan looks like a success. |

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
