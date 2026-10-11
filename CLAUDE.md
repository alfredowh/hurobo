# ego2robo — operating instructions & environment setup

See `PLAN.md` for the full implementation plan and schedule; this file pins the roles, coding standards, and environment so every session uses them consistently.

## Operating mode

Claude works in two roles at once.

### Supervisor
- Before committing to an approach, research existing work (use the browser) on egocentric-data-to-robot retargeting, SmolVLA/VLA post-training, and world models for manipulation. Recommend relevant prior work and state clearly whether the plan: reuses it as-is, modifies it for this challenge's constraints, or departs from it with a novel idea — and why.
- Keep the project on the critical path. Every block in `PLAN.md` has a **gate**. If the gate isn't met by its deadline, take the listed fallback. Don't keep debugging.
- Secure a working deliverable before touching anything ambitious. The order is: MVP replay video, then the generated dataset, then the trained policy, then stretch goals.
- Protect the last few hours before deadline for README and submission, and protect sleep — tired debugging costs more time than it saves.
- Be honest in the write-up. Unverified claims and cherry-picked numbers are out. Judges explicitly penalise "AI slop".

### Developer: coding standards
- **Object-oriented.** Each real entity gets a class with one responsibility (see the repo layout in `PLAN.md`). Scripts are thin: they parse args, build objects, and call methods.
- **Reusable functions.** Logic used twice goes into a shared module: transforms, video I/O, smoothing, resampling, plotting. Never copy-paste between scripts.
- **Documentation.** Each module opens with a one-line docstring. Each public class or function gets a short docstring covering purpose, args, and returns. Comments only explain *why*: sign conventions, frame conventions, assumptions, workarounds.
- **Clean code.** Type hints on signatures. `@dataclass` for data containers such as `HumanDemo` and `CameraIntrinsics`. Descriptive names. No dead code. `ruff format` + `ruff check` before each commit.
- **Frames are explicit.** Every pose variable is named with its frame, e.g. `T_cam_table`, `p_tcp_table`, `T_world_bowl`. Frame bugs are the #1 time-sink in retargeting.
- Working but slightly messy beats clean but late. Refactor only when it's cheap.

## Examples: every feature gets a notebook
- `examples/` holds one Jupyter notebook per capability; `examples/README.md` is the index and lists the conventions.
- **When you implement a new feature, add (or extend) a notebook in `examples/` that showcases it, in the same change.** Prefer interactive output where it helps (e.g. the plotly 3D splat viewer in `01_splat_viewer_3d.ipynb`).
- **Notebooks hold no reusable logic.** A notebook cell is calls into `src/ego2robo` plus at most a few lines of glue (a `print`, a path, a loop over results). Any function, class, plotting helper, setup/bootstrap, or data-building block that a notebook needs, or that a second notebook or script could need, goes into a shared module in the same change. Never define `def`/`class` in a notebook, and never copy a block from one notebook to another.
- Where shared code goes:
  - bootstrap (repo root, `MUJOCO_GL`, input-path checks): `ego2robo.utils.notebook` (`setup_notebook`, `existing_paths`); every notebook's first cell is `ROOT = setup_notebook()`;
  - interactive plotly views (3D, splats, frames, trajectories, aperture): `ego2robo.utils.viewer`; static matplotlib figures (returned as `Figure`, never saved by the helper): `ego2robo.utils.plotting`;
  - stand-in and synthetic data (marker clips, fake signals): `ego2robo.utils.synthetic`;
  - maths on poses and frames: `ego2robo.utils.transforms`;
  - domain logic goes to its domain class/module (`scan`, `splat`, `hand`, `extract`, `human_dataset`, `sim`, ...), not to a util.
- Prefer interactive (plotly) figures over static PNGs for anything spatial or time-series; return the `go.Figure` and call `.show()` in the notebook.
- When a notebook or script needs something that today lives in a script (`scripts/NN_*.py` is not importable), move it into the module and have the script import it too, so there is one definition.
- A new shared function gets a docstring and, if it has logic, a test in `tests/`.
- Execute the notebook (`uv run jupyter nbconvert --to notebook --execute --inplace ...`) and run `ruff format` + `ruff check` on it before committing. Keep stored outputs small. Add its row to the table in `examples/README.md`.
- A feature is not "done" until its notebook runs top to bottom.

## Environment rules
- **uv only.** No conda, no bare `pip install`. Anaconda base is on PATH — always run code via `uv run ...`, never bare `python`.
- **Python 3.12** (required by LeRobot ≥ 0.5).
- **Dependencies via `uv add`.** Commit both `pyproject.toml` and `uv.lock`.
- **PyTorch from the CUDA 12.6 index** (driver is 560.35; cu128/cu130 builds won't see the GPU). Don't upgrade the driver close to the deadline.
- **Only `opencv-contrib-python`** — override out `opencv-python` to avoid the `cv2` module clobber between `hf-libero` and `mediapipe`.
- **Env vars** in `.env`: `MUJOCO_GL=egl` for headless rendering (`glfw` only for a live viewer). `HF_HOME` stays default; already logged in to HF.
- **Never download `HuggingFaceVLA/libero` (70 GB).** Use `lerobot/libero` (1.9 GB) if reference data is needed.
- Large files (raw videos, generated datasets, checkpoints) are never committed. The repo ships scripts plus an HF Hub link for the dataset and model.

## Setup
```bash
cd ~/alfredo/workspace/ego2robo
git init
uv init --package --python 3.12 --name ego2robo
uv python pin 3.12
```

`pyproject.toml` key parts:
```toml
[project]
name = "ego2robo"
requires-python = ">=3.12,<3.13"
dependencies = [
  "lerobot[smolvla,libero,training]>=0.6.1,<0.7",
  "mediapipe",
  "opencv-contrib-python>=4.10",
  "scipy",
  "matplotlib",
]

[dependency-groups]
dev = ["ruff", "ipykernel"]

[tool.uv]
# hf-libero pulls opencv-python; mediapipe pulls opencv-contrib-python. Both install cv2 and clobber each other.
override-dependencies = ["opencv-python; sys_platform == 'never'"]

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"   # driver 560.35 is too old for cu128/cu130
explicit = true
```

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.10/2.11+cu126, True
uv run python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"             # expect True
uv run python -c "import mediapipe, lerobot; print('ok')"
MUJOCO_GL=egl uv run python scripts/00_check_env.py   # builds the LIBERO task env, steps 10 random actions, saves a frame PNG
```

## Known gotchas
- LIBERO's first run may prompt interactively for `~/.libero/config.yaml` if missing — pre-create it if a script hangs.
- Video decoding: if torchcodec errors, pass `--dataset.video_backend=pyav`.
- Pin `lerobot` to whatever version `uv lock` resolves. Don't `uv lock --upgrade` close to the deadline.
