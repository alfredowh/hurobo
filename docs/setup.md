# Hardware, software and workspace


**Capture hardware**

| | |
|---|---|
| Camera | Samsung Galaxy S10e on a tripod (1080p30, stabilization off, focus and exposure locked) |
| Recording trigger | CMF Watch Pro 2 smartwatch, to start and stop the recording |

**Software**

- Python 3.12, managed with **uv only**. Always use `uv run ...`. Do not use bare `python`.
- PyTorch from the CUDA 12.6 index. Only `opencv-contrib-python` is installed. `opencv-python` is overridden out, because it replaces the `cv2` module of `mediapipe`.
- Headless rendering with `MUJOCO_GL=egl`.

```bash
git clone <this repo> && cd hurobo
cp .env.example .env
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect +cu126, True
MUJOCO_GL=egl uv run python scripts/00_check_env.py                                      # builds the LIBERO env, saves a frame
uv run pytest
```

Notes:

- Do not download `HuggingFaceVLA/libero` (70 GB). Use `lerobot/libero` (1.9 GB) if you need reference data.
- If LIBERO asks for `~/.libero/config.yaml` on the first run, create the file before you start.
- If torchcodec fails to decode video, use `--dataset.video_backend=pyav`.
- Marker IDs: ChArUco uses IDs 0 to 16 of `DICT_4X4_50`. Table, bowl and plate markers use IDs 17 or higher.

---

## 8. Workspace structure

```
hurobo/
├── README.md
├── pyproject.toml  uv.lock  ruff.toml  .env.example
├── src/hurobo/
│   ├── camera.py, camera_source.py   # intrinsics, ChArUco calibration, webcam / file / stream sources
│   ├── markers.py                    # ArUco poses: T_cam_table, T_table_bowl, T_table_plate
│   ├── hand.py, live_hand.py         # MediaPipe + PnP hand TCP and pinch aperture (offline / live)
│   ├── demo.py, extract.py           # HumanDemo, grasp segmentation, DemoExtractor + QC report
│   ├── human_dataset.py, recorder.py # HumanDemo → LeRobotDataset v3
│   ├── twin.py, sim.py               # SceneTwin, LIBERO wrapper
│   ├── control.py                    # waypoints → clipped 7-D delta actions
│   ├── teleop.py, teleop_app.py, live.py, props.py   # hand/keyboard teleop of the Panda, live window, grasp cube
│   ├── scan.py, scan_sweep.py, splat.py, collider.py # orbit video → splat → composite → collision mesh; scan parameter sweep
│   ├── real_setup.py, object_scale.py, scaled_native.py # measurements.yaml, MJCF scaling, E1 scene (scaled bowl and plate)
│   ├── capture_plan.py, recording.py # phone pose search, layouts, markers, capture check
│   ├── retarget.py, generate.py, reach.py # human demo → robot demo from home, dataset generation, reach map
│   ├── mimicgen.py                   # MimicGen-style expansion: nearest-layout sources, action noise
│   ├── clip_clean.py, clip_cutter_app.py # cleaning of real clips, manual clip cutter
│   ├── training.py, checkpoint_eval.py, model_comparison.py, policy_rollout.py, benchmark.py # SmolVLA fine-tune, per-checkpoint and cross-model evaluation
│   └── utils/                        # transforms, signal, video, plotting, viewer, synthetic, notebook
├── scripts/                          # thin CLIs: NN_name.py
├── examples/                         # one notebook for each capability (see notebooks.md)
├── docs/                             # scanning_guide.md, camera_sources.md, design_decisions.md
├── tests/                            # pytest
└── data/                             # not in git: raw clips, intrinsics, scans, splats, datasets
```

Rules for the code:

- Each module has one responsibility. Scripts only parse arguments and call objects.
- Each pose variable has its frame in its name (`T_cam_table`, `p_tcp_table`).
