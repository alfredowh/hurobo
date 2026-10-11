"""Bootstrap shared by every `examples/` notebook: project root, headless GL, input-path checks."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Nearest ancestor of `start` (default: the working directory) that holds this package's source.

    Args:
        start: where to begin the upward search.

    Raises:
        RuntimeError: if no ancestor looks like the repo root.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src/hurobo").is_dir():
            return candidate
    raise RuntimeError(f"no hurobo checkout found above {here}")


def setup_notebook(mujoco_gl: str = "egl") -> Path:
    """Make a notebook start from the repo root with headless MuJoCo rendering, and return that root.

    Notebooks live in `examples/` while `data/` and `scripts/` are one level up, so
    relative paths only resolve after the chdir. `MUJOCO_GL` must be set before
    mujoco is first imported, hence call this in the first cell.

    Args:
        mujoco_gl: rendering backend; `egl` is headless, `glfw` needs a display.

    Returns:
        The repo root.
    """
    root = find_project_root()
    os.chdir(root)
    os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    return root


def existing_paths(paths: dict[str, Path], hint: str) -> dict[str, Path]:
    """Keep only the entries whose file exists; fail with `hint` if none do.

    Args:
        paths: display name -> file path.
        hint: what to run to produce the missing inputs.

    Raises:
        FileNotFoundError: if no path exists.
    """
    found = {name: path for name, path in paths.items() if path.exists()}
    if not found:
        raise FileNotFoundError(f"none of the expected inputs exist; {hint}")
    return found


def show_video(video_path: Path, width: int = 640):
    """Embed a video in the notebook output (looping, muted, with controls).

    The file is embedded as base64, so the notebook plays it without the
    original path; keep clips short.

    Args:
        video_path: .mp4 to embed.
        width: display width in pixels.
    """
    from IPython.display import Video

    return Video(str(video_path), embed=True, width=width, html_attributes="controls loop autoplay muted")
