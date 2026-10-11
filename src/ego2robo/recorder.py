"""Schema-agnostic LeRobotDataset v3 writer shared by the human-demo export and the sim generator."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SIDECAR_NAME = "ego2robo_meta.json"


def video_feature(height: int, width: int) -> dict[str, Any]:
    """Feature spec for one RGB camera stream stored as video.

    Args:
        height: image height in pixels.
        width: image width in pixels.

    Returns:
        A LeRobot feature dict (dtype "video", HWC).
    """
    return {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"]}


def vector_feature(names: list[str]) -> dict[str, Any]:
    """Feature spec for a float32 vector with one name per component.

    Args:
        names: component names; their count is the vector length.

    Returns:
        A LeRobot feature dict (dtype "float32", shape (len(names),)).
    """
    return {"dtype": "float32", "shape": (len(names),), "names": names}


class LeRobotRecorder:
    """Writes episodes of frame dicts into a new local LeRobotDataset.

    Thin on purpose: it owns create / add-episode / finalize and a metadata
    sidecar, and knows nothing about what the features mean. Callers supply the
    schema (`features`) and per-frame dicts matching it, which keeps the same
    class usable for human demos now and sim-generated demos later.
    """

    def __init__(
        self,
        root: Path,
        repo_id: str,
        fps: int,
        features: dict[str, dict[str, Any]],
        robot_type: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Create an empty dataset on disk.

        Args:
            root: dataset directory; must not exist unless `overwrite` is set.
            repo_id: Hub-style id (`user/name`); only used for metadata until pushed.
            fps: sampling rate of every frame; LeRobot derives timestamps as index / fps.
            features: feature spec, e.g. built with `video_feature` / `vector_feature`.
            robot_type: free-form string stored in `info.json`.
            overwrite: delete an existing `root` first.

        Raises:
            FileExistsError: if `root` exists and `overwrite` is False.
        """
        if root.exists():
            if not overwrite:
                raise FileExistsError(f"{root} exists; pass overwrite=True to replace it")
            shutil.rmtree(root)
        self._root = root
        self._dataset = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, features=features, root=root, robot_type=robot_type, use_videos=True
        )
        self._num_episodes = 0

    @property
    def root(self) -> Path:
        """Directory the dataset is written to."""
        return self._root

    def add_episode(self, frames: Iterable[dict[str, Any]], task: str) -> int:
        """Append one episode.

        Args:
            frames: per-frame dicts keyed by feature name (images as HxWx3 uint8 RGB,
                vectors as float32 arrays). Must not contain `task` or LeRobot's
                automatic keys (timestamp, frame_index, ...).
            task: natural-language instruction, stored once per episode.

        Returns:
            The episode's index in the dataset.

        Raises:
            ValueError: if `frames` is empty.
        """
        count = 0
        for frame in frames:
            self._dataset.add_frame({**frame, "task": task})
            count += 1
        if count == 0:
            self._dataset.clear_episode_buffer()
            raise ValueError("an episode needs at least one frame")
        self._dataset.save_episode()
        self._num_episodes += 1
        return self._num_episodes - 1

    def write_sidecar(self, metadata: dict[str, Any]) -> None:
        """Write ego2robo-specific metadata (calibration, thresholds, provenance) next to the data.

        Args:
            metadata: JSON-serialisable dict, saved as `ego2robo_meta.json` in the root.
        """
        (self._root / SIDECAR_NAME).write_text(json.dumps(metadata, indent=2))

    def finalize(self) -> None:
        """Flush writers. Required: without it the parquet footers are missing and the dataset is invalid."""
        self._dataset.finalize()

    def __enter__(self) -> LeRobotRecorder:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.finalize()
