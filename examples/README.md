# Examples

One notebook per capability of this workspace. Each is self-contained, runs top to bottom, and ships with its outputs so it can be read without running anything.

| Notebook | Shows | Needs |
|---|---|---|
| [01_splats_and_scans](01_splats_and_scans.ipynb) | Splats end to end: interactive 3D view and rendered orbits; scan pipeline (orbit video → COLMAP cameras → object frame, best scans, photo vs splat); splats composited into LIBERO by depth | `data/splats/`, `data/scans/`; GPU + EGL for the composite part |
| [02_frames_and_calibration](02_frames_and_calibration.ipynb) | Frame conventions, camera intrinsics → MuJoCo FOV, ChArUco board, true-size on-screen marker sheet for a tablet, grasp segmentation | nothing |
| [03_hand_tracking](03_hand_tracking.ipynb) | MediaPipe + PnP hand TCP (interactive per-frame curves, overlay video, whole-hand overlay, interactive pinch-aperture step-through, synced overlay + 3D hand + aperture video), ArUco table/bowl/plate poses vs ground truth, `DemoExtractor` QC report | `data/raw/overview_gif/first_clip/` (stand-in clip) |
| [04_lerobot_recording](04_lerobot_recording.ipynb) | `HumanDemo` → LeRobotDataset v3 export and reload (schema, video, state/action, state-vs-action overlay, sidecar, synced episode video) | same stand-in clip |
| [05_teleop](05_teleop.ipynb) | Hand-clip teleoperation of the LIBERO Panda: table-frame hand path under two camera poses, table-to-world mapping, tracking error, cube spawned on grasp, 1x and slowed-down videos, scripted keyboard driving | GPU/EGL, `data/raw/overview_gif/hand_moves_cube/` (stand-in clip; markers are synthetic) |
| [06_scan_parameter_sweep](06_scan_parameter_sweep.ipynb) | How COLMAP/SIFT settings, matching mode, keyframe spacing and training length change a phone scan (registration, orbit coverage, held-out PSNR, recovered cameras, held-out photo vs splat), and three ways to crop the object (current percentile fit, RANSAC floor + cluster, explicit) against an independent reference. Part 2: the bowl from three heights, the plate and clip combinations (registration, objects alone, white-object ray-distance consistency check, candidates composited in LIBERO). Part 5: the plate completed with a measured shape prior (`DishFiller`) against the scan. Part 4: the table and background with the ChArUco board on a tablet (board visibility per clip, two clips in one model, tabletop box, plate and bowl cut out) | `data/scans/table_zoom_02_03/`, `data/splats/table_zoom_02_03_*.ply` (from `scripts/09_scan_object.py --charuco`), `data/scans/sweeps/bowl_1angle/` (from `scripts/14_scan_param_sweep.py`) and `data/scans/clips/` (from `scripts/15_scan_clips.py`); real phone clips, not committed, GPU/EGL |
| [07_whole_scene](07_whole_scene.ipynb) | The whole `table_zoom_02_03` scan (table, plate, bowl, background) merged back into one splat (`merge_plys`), viewed as interactive 3D Gaussian centres with per-layer switching, and as photo \| splat renders from the phone's own camera path in the table frame | `data/splats/table_zoom_02_03_*.ply` and `data/scans/table_zoom_02_03/` (from `scripts/09_scan_object.py --charuco`), GPU |
| [08_scaled_native_experiment](08_scaled_native_experiment.ipynb) | Experiment E1 (real-size bowl and plate in LIBERO's own scene): measured vs LIBERO sizes and the scaling, why the other objects are parked, the Panda's reach map, the recording plan (phone pose search, layouts and hand starts, bowl pad), the command list, human demo to robot demo from the home pose (twin replays, trajectories, videos) and the success rate of retargeted rollouts. **The hand input is `synthetic_pick_place_demo`**; the plan is built from the real measurements | `data/raw/setup_01/measurements.yaml`, `data/camera_intrinsics_s24.npz`, GPU/EGL |
| [09_smolvla_benchmark](09_smolvla_benchmark.ipynb) | SmolVLA benchmark in LIBERO: the two most-liked LIBERO checkpoints in three scene conditions (stock, cleared table, cleared table with real-size bowl and plate), success with Wilson intervals, how far episodes got, open-loop action error against demonstrations, rollout videos | `eval/pretrained_libero_checkpoints/` (from `scripts/18_benchmark_policy.py`), GPU/EGL |
| [10_clip_cleaning](10_clip_cleaning.ipynb) | The cleaning step before extraction: per-frame filters (no hand, two hands, bad fit, image edge, depth), the cut to one open-close-open cycle with a margin after the release, clip filters, the per-clip timeline and the proposed `cuts.json`; how to adjust the cuts in the clip cutter UI | `data/raw/e1/test_0*.mp4` (real phone test clips, not committed), `data/camera_intrinsics_s10e.npz` |
| [11_recorded_dataset_and_replay](11_recorded_dataset_and_replay.ipynb) | A cleaned real clip end to end: overlay video (markers, hand skeleton, projected TCP and path, gripper symbol), the LeRobot episode dashboard, interactive TCP/aperture, the human's bowl/plate layout in the sim (`layout_of_demo`, clamping), the robot replay in the scaled LIBERO scene, and dataset and replay side by side | `data/raw/e1/test_03.mp4`, `data/e1/demos`, `data/e1/human` (not committed), `data/camera_intrinsics_s10e.npz` |
| [14_smolvla_model_comparison](14_smolvla_model_comparison.ipynb) | Best checkpoint of our fine-tune from `smolvla_base`, our fine-tune from `lerobot/smolvla_libero`, `HuggingFaceVLA/smolvla_libero` and `lerobot/smolvla_libero` on the same scaled scene and held-out layouts: closed-loop success with intervals, open-loop error, one video each | `outputs/e1_smolvla_*/run`, `results/smolvla_*ckpts`, `data/e1/generated_clips` (not committed) |
| [12_mimicgen_expansion](12_mimicgen_expansion.ipynb) | MimicGen-style expansion of the usable real clips (`hurobo.mimicgen`): nearest-layout source choice per subtask (interactive view), mixed approach/carry plans, action noise, source screening and a 300-step filter; stock `scripts/04` rollouts against the new ones on the same layouts, and the stored longer runs (success, failure reasons, steps, per-source usage) | `data/e1/demos_clips` (real clips, not committed; stand-in demos as fallback), `data/raw/setup_01/measurements.yaml`, runs of `scripts/22_mimicgen_expand.py` in `data/e1/mimicgen/` for section 3 |
| [13_smolvla_checkpoints](13_smolvla_checkpoints.ipynb) | SmolVLA fine-tuned on the generated robot dataset, evaluated at every saved checkpoint: closed-loop success in the scaled scene (Wilson intervals, failure reasons, n per point), open-loop chunk error against the demonstrations (training frames), metrics against training step, training loss, rollout videos. Shows whatever checkpoints exist when executed | `outputs/e1_smolvla_ckpts/` (from `scripts/19_train_smolvla.py`), `results/smolvla_ckpts/` (from `scripts/23_eval_checkpoints.py`), GPU/EGL |

Camera sources for live teleoperation (webcam, phone, streams) are documented in [`docs/camera_sources.md`](../docs/camera_sources.md).

The `data/` inputs are produced by `scripts/08_splat_composite.py`, `scripts/09_scan_object.py`, `scripts/14_scan_param_sweep.py` and `scripts/15_scan_clips.py` (they are not committed).

## Running

Open in VS Code (kernel: the project's `.venv`) or:

```bash
uv run jupyter lab examples/
```

Re-execute one in place (this is how the stored outputs were produced):

```bash
MUJOCO_GL=egl uv run jupyter nbconvert --to notebook --execute --inplace examples/01_splats_and_scans.ipynb
```

## Conventions for new notebooks

- Name `NN_topic.ipynb`, numbered in order; add a row to the table above.
- First cell: `%load_ext autoreload` + `%autoreload 2`, so edits to `src/hurobo` reload without a kernel restart.
- Second cell: `from hurobo.utils.notebook import setup_notebook` then `ROOT = setup_notebook()` (repo root, headless GL).
- Import from `src/hurobo`; never define a function or class in a notebook and never copy code between notebooks. Anything a cell needs beyond a few lines of glue goes into a shared module in the same change:
  - interactive plotly views: `hurobo.utils.viewer` (return a `go.Figure`, call `.show()` in the notebook);
  - static matplotlib figures: `hurobo.utils.plotting` (return a `Figure`);
  - stand-in/synthetic inputs: `hurobo.utils.synthetic` (`StandInScene`, `synthetic_aperture_trace`);
  - pose maths: `hurobo.utils.transforms`; everything else in its domain module.
- Say in the intro which module it demonstrates and what is stand-in or synthetic data.
- Execute before committing, and keep stored outputs small (subsample point clouds, downscale images).
