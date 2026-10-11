#!/usr/bin/env python
"""Sanity-checks the environment: GPU, cv2.aruco, mediapipe, and a LIBERO render.

Run with: MUJOCO_GL=egl uv run python scripts/00_check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def check_torch() -> bool:
    """Print torch's version/build and whether the GPU is visible."""
    import torch

    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
    return torch.cuda.is_available()


def check_cv2() -> bool:
    """Print cv2's version and confirm the aruco module is present (contrib build)."""
    import cv2

    has_aruco = hasattr(cv2, "aruco")
    print(f"cv2 {cv2.__version__}, has aruco: {has_aruco}")
    return has_aruco


def check_mediapipe() -> bool:
    """Confirm mediapipe imports and the Tasks HandLandmarker API is reachable."""
    import mediapipe as mp

    has_tasks = hasattr(mp, "tasks") and hasattr(mp.tasks.vision, "HandLandmarker")
    print(f"mediapipe {mp.__version__}, has Tasks HandLandmarker: {has_tasks}")
    return has_tasks


def check_libero_render() -> bool:
    """Build the target LIBERO task (libero_goal, task 8: "put the bowl on the
    plate"), step with the controller's no-op action, save a frame PNG.

    This is the real gate (PLAN.md §2): if this fails, fall back to plain
    robosuite per the plan's risk register.
    """
    from PIL import Image

    from hurobo.sim import LiberoTask

    try:
        task = LiberoTask(suite_name="libero_goal", task_id=8)
    except Exception as exc:  # noqa: BLE001 - diagnostic script, report and continue
        print(f"Could not build the LIBERO env: {exc}")
        return False

    obs = task.reset()
    print(f"Task description: {task.task_description!r}")
    no_op_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(10):
        obs, _is_success, _terminated = task.step(no_op_action)

    out_path = Path("data/check_env_frame.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_array = np.asarray(obs["pixels"]["image"])[0]
    Image.fromarray(image_array).save(out_path)
    print(f"Saved a frame to {out_path}")
    task.close()
    return True


def main() -> int:
    """Run every check and exit non-zero if any critical one fails."""
    results = {
        "torch/cuda": check_torch(),
        "cv2.aruco": check_cv2(),
        "mediapipe": check_mediapipe(),
        "libero_render": check_libero_render(),
    }
    print("\nSummary:")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
