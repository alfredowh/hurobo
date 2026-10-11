"""Live camera sources (webcams, phone bridges, network streams, files) behind one interface."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

SourceKind = Literal["frames", "video", "webcam", "stream", "file"]

_STREAM_SCHEMES = ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")


@dataclass(frozen=True)
class SourceSpec:
    """A parsed camera-source selector.

    Attributes:
        kind: `frames` (folder of images) and `video` (file) are replayed offline; `webcam`
            (V4L2 index or path), `stream` (network URL) and `file` (a video file played back
            in real time as if it were a camera) are live.
        target: device index or path, URL, or file/folder path.
    """

    kind: SourceKind
    target: str | int

    @property
    def is_live(self) -> bool:
        """True for sources that are read frame by frame as they arrive."""
        return self.kind in ("webcam", "stream", "file")


def parse_source(text: str) -> SourceSpec:
    """Interpret a source string given on the command line.

    Forms: `webcam`, `webcam:2`, `webcam:/dev/video2`, `2`, `/dev/video2` (V4L2 device);
    `http://...`, `rtsp://...` or `url:...` (network stream); `file:clip.mp4` (a video file
    played in real time through the live pipeline); a folder of images (`frames`) or a video
    file (`video`) is replayed offline.

    Args:
        text: the selector.

    Raises:
        ValueError: if it matches none of the forms.
    """
    text = text.strip()
    if text == "webcam":
        return SourceSpec("webcam", 0)
    if text.startswith("webcam:"):
        return SourceSpec("webcam", _device(text.removeprefix("webcam:")))
    if text.startswith("url:"):
        return SourceSpec("stream", text.removeprefix("url:"))
    if text.lower().startswith(_STREAM_SCHEMES):
        return SourceSpec("stream", text)
    if text.startswith("file:"):
        return SourceSpec("file", text.removeprefix("file:"))
    if text.isdigit() or text.startswith("/dev/video"):
        return SourceSpec("webcam", _device(text))
    path = Path(text)
    if path.is_dir():
        return SourceSpec("frames", text)
    if path.is_file():
        return SourceSpec("video", text)
    raise ValueError(
        f"cannot interpret source {text!r}: expected an image folder, a video file, 'webcam[:N]', "
        "'/dev/videoN', a URL (http/rtsp), or 'file:<video>'"
    )


def _device(text: str) -> str | int:
    return int(text) if text.isdigit() else text


class FrameSource(Protocol):
    """What the live pipeline needs from a camera."""

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the frames."""
        ...

    @property
    def ended(self) -> bool:
        """True once no further frames will arrive."""
        ...

    def latest(self) -> tuple[np.ndarray, int] | None:
        """Newest BGR frame and its running index (None before the first frame arrives)."""
        ...

    def close(self) -> None:
        """Release the device."""
        ...


def _open_stream(url: str) -> cv2.VideoCapture:
    """Open a network stream with low-latency FFmpeg options, applied to this capture only.

    The options travel in an environment variable that OpenCV reads when a capture is created;
    leaving it set would also change how every later `cv2.VideoCapture` opens ordinary files,
    so it is restored straight after the open.
    """
    options = ["fflags;nobuffer", "flags;low_delay"]
    if url.lower().startswith("rtsp://"):
        options.insert(0, "rtsp_transport;tcp")  # UDP drops packets on Wi-Fi
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    previous = os.environ.get(key)
    os.environ[key] = "|".join(options)
    try:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    finally:
        if previous is None:
            del os.environ[key]
        else:
            os.environ[key] = previous


class CaptureFrameSource:
    """A `cv2.VideoCapture` read continuously on a thread, so `latest` always returns the newest frame.

    Reading on demand from a camera or network stream returns old buffered frames and shows up as
    seconds of lag; draining on a thread and keeping only the last frame removes that.
    """

    def __init__(
        self,
        spec: SourceSpec,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        open_timeout_s: float = 10.0,
    ) -> None:
        """Open the device or stream and start reading.

        Args:
            spec: a `webcam` or `stream` spec.
            width: requested frame width (webcams only; the driver may pick the nearest mode).
            height: requested frame height.
            fps: requested frame rate (webcams only).
            open_timeout_s: how long to wait for the first frame.

        Raises:
            RuntimeError: if the source cannot be opened or never delivers a frame.
        """
        if spec.kind == "webcam":
            capture = cv2.VideoCapture(spec.target, cv2.CAP_V4L2)
            # MJPG: uncompressed YUYV over USB 2 caps at low resolutions / frame rates.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            if width and height:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                capture.set(cv2.CAP_PROP_FPS, fps)
        elif spec.kind == "stream":
            capture = _open_stream(str(spec.target))
        else:
            raise ValueError(f"CaptureFrameSource handles webcam and stream sources, not {spec.kind!r}")
        if not capture.isOpened():
            raise RuntimeError(f"could not open camera source {spec.target!r}")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._index = 0
        self._ended = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + open_timeout_s
        while self.latest() is None:
            if self._ended or time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"camera source {spec.target!r} opened but delivered no frames")
            time.sleep(0.02)
        self._size = (self._frame.shape[1], self._frame.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        """See `FrameSource`."""
        return self._size

    @property
    def ended(self) -> bool:
        """See `FrameSource`."""
        return self._ended

    def latest(self) -> tuple[np.ndarray, int] | None:
        """See `FrameSource`."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._index

    def close(self) -> None:
        """Stop the reader thread and release the device. Safe to call twice."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self._capture.release()

    def _read_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                failures += 1
                if failures > 50:  # ~ a second of consecutive failed reads: the stream is gone
                    self._ended = True
                    return
                time.sleep(0.02)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self._index += 1


class FileFrameSource:
    """A video file presented as a camera, so the live pipeline can be tried (and tested) without hardware.

    Frames advance with calls to `latest`, one per call, not with the wall clock: the
    pipeline's pacing (one query per sim step) then decides playback speed, and a test is
    deterministic. It ends after the last frame.
    """

    def __init__(self, path: Path, loop: bool = False) -> None:
        """Decode the whole (short) clip.

        Args:
            path: video file.
            loop: restart from the first frame instead of ending.
        """
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video {path}")
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames:
            raise RuntimeError(f"{path} has no frames")
        self._frames = frames
        self._loop = loop
        self._next = 0
        self._served = 0

    @property
    def size(self) -> tuple[int, int]:
        """See `FrameSource`."""
        return self._frames[0].shape[1], self._frames[0].shape[0]

    @property
    def ended(self) -> bool:
        """See `FrameSource`."""
        return not self._loop and self._next >= len(self._frames)

    def latest(self) -> tuple[np.ndarray, int] | None:
        """See `FrameSource`; each call consumes one frame (the last one is repeated at the end)."""
        index = self._next % len(self._frames) if self._loop else min(self._next, len(self._frames) - 1)
        self._next += 1
        self._served += 1
        return self._frames[index], self._served

    def close(self) -> None:
        """See `FrameSource`."""


class MirroredFrameSource:
    """Flips every frame left-right, so a webcam behaves like a mirror (the convention people expect).

    The flip is applied *before* tracking, so the metric hand pose and the picture stay
    consistent (a reflection of the camera's x axis), and moving your right hand to the right
    moves the robot to the right of the agentview picture.
    """

    def __init__(self, inner: FrameSource) -> None:
        """Wrap another source."""
        self._inner = inner

    @property
    def size(self) -> tuple[int, int]:
        """See `FrameSource`."""
        return self._inner.size

    @property
    def ended(self) -> bool:
        """See `FrameSource`."""
        return self._inner.ended

    def latest(self) -> tuple[np.ndarray, int] | None:
        """See `FrameSource`."""
        newest = self._inner.latest()
        if newest is None:
            return None
        return np.ascontiguousarray(newest[0][:, ::-1]), newest[1]

    def close(self) -> None:
        """See `FrameSource`."""
        self._inner.close()


def open_frame_source(
    spec: SourceSpec, width: int | None = None, height: int | None = None, fps: float | None = None
) -> FrameSource:
    """Open a live source.

    Args:
        spec: a `webcam`, `stream` or `file` spec.
        width: requested width (webcams).
        height: requested height (webcams).
        fps: requested frame rate (webcams).

    Raises:
        ValueError: for offline kinds (`frames`, `video`).
    """
    if spec.kind == "file":
        return FileFrameSource(Path(str(spec.target)))
    if spec.kind in ("webcam", "stream"):
        return CaptureFrameSource(spec, width, height, fps)
    raise ValueError(f"{spec.kind!r} sources are replayed offline, not opened live")


@dataclass(frozen=True)
class CameraInfo:
    """One V4L2 node found on this machine.

    Attributes:
        device: path such as `/dev/video0`; its number is the index OpenCV expects.
        name: driver-reported name.
        is_capture_node: False for metadata nodes that share a physical camera.
    """

    device: str
    name: str
    is_capture_node: bool


def list_cameras(sysfs_root: Path = Path("/sys/class/video4linux")) -> list[CameraInfo]:
    """Enumerate V4L2 devices from sysfs without opening them (so no camera LED lights up).

    Phone bridges (scrcpy + v4l2loopback, DroidCam, a UVC-mode phone) appear here too, as
    soon as the bridge is running.

    Args:
        sysfs_root: where the kernel lists video nodes; overridable for tests.
    """
    cameras = []
    for node in sorted(sysfs_root.glob("video*"), key=lambda p: int(p.name.removeprefix("video"))):
        name = (node / "name").read_text().strip() if (node / "name").exists() else node.name
        index_file = node / "index"
        is_capture = not index_file.exists() or index_file.read_text().strip() == "0"
        cameras.append(CameraInfo(f"/dev/{node.name}", name, is_capture))
    return cameras
