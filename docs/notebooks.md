# Notebooks


The notebooks are in [examples/](../examples/). Each notebook runs from top to bottom and stores its outputs. Each notebook calls `src/hurobo` and contains no reusable logic.

| Notebook | Content |
|---|---|
| [01_splats_and_scans](../examples/01_splats_and_scans.ipynb) | Splats end to end: interactive 3D view, scan pipeline (orbit video → COLMAP → object frame, best scans), splats composited into LIBERO by depth. |
| [02_frames_and_calibration](../examples/02_frames_and_calibration.ipynb) | Frame conventions, intrinsics → MuJoCo FOV, ChArUco board, on-screen marker sheet, grasp segmentation. |
| [03_hand_tracking](../examples/03_hand_tracking.ipynb) | MediaPipe + PnP hand TCP, pinch-aperture step-through, ArUco poses against ground truth, `DemoExtractor` QC report. |
| [04_lerobot_recording](../examples/04_lerobot_recording.ipynb) | `HumanDemo` → LeRobotDataset v3 export and reload: schema, video, state and action, sidecar file. |
| [05_teleop](../examples/05_teleop.ipynb) | Hand-clip teleoperation of the LIBERO Panda: table-to-world mapping, tracking error, cube spawned on grasp, scripted keyboard control. |
| [06_scan_parameter_sweep](../examples/06_scan_parameter_sweep.ipynb) | How SfM settings, matching mode, keyframe spacing and training length change a real phone scan (registration, orbit coverage, held-out PSNR), and three ways of cropping the object compared against an independent reference. Real clip, not committed. |
| [07_whole_scene](../examples/07_whole_scene.ipynb) | The whole table scan merged into one splat: interactive 3D view with a layer switch, photo and splat renders from the phone's camera path, a browser viewer with a crop box, still 3D views of the cut scene and the README sweep. |
| [08_scaled_native_experiment](../examples/08_scaled_native_experiment.ipynb) | Experiment E1: real-size bowl and plate in LIBERO, the recording plan (phone pose, layouts, markers, commands), human demo to robot demo from the home pose, success rate of retargeted rollouts. Hand input is made up. |
| [09_smolvla_benchmark](../examples/09_smolvla_benchmark.ipynb) | SmolVLA benchmark in LIBERO: the two most-liked LIBERO checkpoints in three scene conditions (stock, cleared table, cleared table with real-size bowl and plate), success with Wilson intervals, how far episodes got, open-loop action error, rollout videos. |
| [10_clip_cleaning](../examples/10_clip_cleaning.ipynb) | Cleaning real phone clips before extraction: per-frame filters (no hand, two hands, bad fit, image edge, depth), the cut to one open-close-open cycle, clip filters, the per-clip timeline and the proposed `cuts.json`. Real clips, not committed. |
| [11_recorded_dataset_and_replay](../examples/11_recorded_dataset_and_replay.ipynb) | One cleaned real clip end to end: overlay video, LeRobot episode dashboard, interactive TCP and aperture, the human's bowl and plate layout in the sim, robot replay in the scaled LIBERO scene, dataset and replay side by side. Real clip, not committed. |
| [12_mimicgen_expansion](../examples/12_mimicgen_expansion.ipynb) | Alternative demo generation (MimicGen-style) against the default `DemoGenerator` on the same layouts: source choice, screening, action noise, step filter; the longer runs behind the optimization-potential result. |
| [13_smolvla_checkpoints](../examples/13_smolvla_checkpoints.ipynb) | SmolVLA fine-tuned on the generated dataset, evaluated at every saved checkpoint: closed-loop success (Wilson intervals, failure reasons), open-loop action error on training frames, training loss, rollout videos. |
| [14_smolvla_model_comparison](../examples/14_smolvla_model_comparison.ipynb) | Our fine-tuned policies against the two pretrained LIBERO checkpoints on the same scaled scene and held-out layouts. |

Notebooks 01 to 05 and 08 use **stand-in data** (dummy scans, a stand-in hand clip with synthetic markers). Notebooks 06, 10, 11 and 12 read real phone clips (not committed); 09, 13 and 14 read benchmark and evaluation results, 13 and 14 for the policy trained on the generated data. Each notebook states this in its introduction.

```bash
uv run jupyter lab examples/
MUJOCO_GL=egl uv run jupyter nbconvert --to notebook --execute --inplace examples/01_splats_and_scans.ipynb
