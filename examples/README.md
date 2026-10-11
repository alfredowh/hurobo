# Examples

One notebook per capability of this workspace. Each is self-contained, runs top to bottom, and ships with its outputs so it can be read without running anything.

| Notebook | Shows | Needs |
|---|---|---|
| [01_splat_viewer_3d](01_splat_viewer_3d.ipynb) | Interactive 3D view of Gaussian-splat `.ply` files (own scans and DISCOVERSE assets) | `data/splats/` |
| [02_scan_pipeline](02_scan_pipeline.ipynb) | Orbit video → COLMAP cameras and sparse points → object frame; photo vs splat check | `data/scans/dummy_fruit_bowl/` |
| [03_splat_composite](03_splat_composite.ipynb) | Splats composited into LIBERO by depth: layers, and splats following moved bodies | GPU, EGL (DISCOVERSE stand-in splats are downloaded on first run into `data/splats/dummy_discoverse/`) |
| [04_frames_and_calibration](04_frames_and_calibration.ipynb) | Frame conventions, camera intrinsics → MuJoCo FOV, ChArUco board, grasp segmentation | nothing |
| [05_hand_tracking](05_hand_tracking.ipynb) | MediaPipe + PnP hand TCP (interactive per-frame curves, overlay video, whole-hand overlay, interactive pinch-aperture step-through, synced overlay + 3D hand + aperture video), ArUco table/bowl/plate poses vs ground truth, `DemoExtractor` QC report | `data/raw/overview_gif/first_clip/` (stand-in clip) |
| [06_lerobot_recording](06_lerobot_recording.ipynb) | `HumanDemo` → LeRobotDataset v3 export and reload (schema, video, state/action, sidecar, synced episode video) | same stand-in clip |

The `data/` inputs are produced by `scripts/08_splat_composite.py` and `scripts/09_scan_object.py` (they are not committed).

## Running

Open in VS Code (kernel: the project's `.venv`) or:

```bash
uv run jupyter lab examples/
```

Re-execute one in place (this is how the stored outputs were produced):

```bash
MUJOCO_GL=egl uv run jupyter nbconvert --to notebook --execute --inplace examples/01_splat_viewer_3d.ipynb
```

## Conventions for new notebooks

- Name `NN_topic.ipynb`, numbered in order; add a row to the table above.
- First cell: `from ego2robo.utils.notebook import setup_notebook` then `ROOT = setup_notebook()` (repo root, headless GL).
- Import from `src/ego2robo`; never define a function or class in a notebook and never copy code between notebooks. Anything a cell needs beyond a few lines of glue goes into a shared module in the same change:
  - interactive plotly views: `ego2robo.utils.viewer` (return a `go.Figure`, call `.show()` in the notebook);
  - static matplotlib figures: `ego2robo.utils.plotting` (return a `Figure`);
  - stand-in/synthetic inputs: `ego2robo.utils.synthetic` (`StandInScene`, `synthetic_aperture_trace`);
  - pose maths: `ego2robo.utils.transforms`; everything else in its domain module.
- Say in the intro which module it demonstrates and what is stand-in or synthetic data.
- Execute before committing, and keep stored outputs small (subsample point clouds, downscale images).
