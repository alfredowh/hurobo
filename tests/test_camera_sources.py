"""Camera-source parsing and capture, online hand tracking, and the live command source."""

from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hurobo.camera_source import (
    CaptureFrameSource,
    FileFrameSource,
    MirroredFrameSource,
    SourceSpec,
    list_cameras,
    open_frame_source,
    parse_source,
)
from hurobo.demo import GraspSegmenter
from hurobo.hand import HandFrameResult
from hurobo.live_hand import LiveHandCommandSource, LiveHandTracker
from hurobo.teleop import ScriptedKeys, TableToWorld
from hurobo.utils.signal import ExponentialSmoother
from hurobo.utils.transforms import level_camera_rotation, make_transform
from hurobo.utils.video import VideoWriter


def test_parse_source_forms(tmp_path: Path) -> None:
    folder, video = tmp_path / "frames", tmp_path / "clip.mp4"
    folder.mkdir()
    video.write_bytes(b"x")
    assert parse_source("webcam") == SourceSpec("webcam", 0)
    assert parse_source("webcam:3") == SourceSpec("webcam", 3)
    assert parse_source("2") == SourceSpec("webcam", 2)
    assert parse_source("/dev/video4") == SourceSpec("webcam", "/dev/video4")
    assert parse_source("webcam:/dev/video4") == SourceSpec("webcam", "/dev/video4")
    assert parse_source("http://192.168.1.5:8080/video") == SourceSpec(
        "stream", "http://192.168.1.5:8080/video"
    )
    assert parse_source("rtsp://host/stream").kind == "stream"
    assert parse_source("url:http://x/y") == SourceSpec("stream", "http://x/y")
    assert parse_source(f"file:{video}") == SourceSpec("file", str(video))
    assert parse_source(str(folder)).kind == "frames" and not parse_source(str(folder)).is_live
    assert parse_source(str(video)).kind == "video" and not parse_source(str(video)).is_live
    assert parse_source("webcam:0").is_live and parse_source("http://x").is_live
    with pytest.raises(ValueError, match="cannot interpret"):
        parse_source("nonsense")


def test_offline_kinds_are_not_opened_live() -> None:
    with pytest.raises(ValueError):
        open_frame_source(SourceSpec("video", "x.mp4"))


def test_list_cameras_reads_sysfs_without_opening(tmp_path: Path) -> None:
    for number, name, index in [(0, "Cam", "0"), (1, "Cam", "1"), (10, "Loop", "0")]:
        node = tmp_path / f"video{number}"
        node.mkdir()
        (node / "name").write_text(name + "\n")
        (node / "index").write_text(index + "\n")
    cameras = list_cameras(tmp_path)
    assert [c.device for c in cameras] == ["/dev/video0", "/dev/video1", "/dev/video10"]  # numeric order
    assert [c.is_capture_node for c in cameras] == [True, False, True]


def write_clip(path: Path, num: int = 6) -> None:
    with VideoWriter(path, 10.0, (64, 48)) as writer:
        for k in range(num):
            writer.write(np.full((48, 64, 3), 10 * k, np.uint8))


def test_file_source_steps_per_query_and_ends(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    write_clip(clip, 4)
    source = FileFrameSource(clip)
    assert source.size == (64, 48) and not source.ended
    indices = [source.latest()[1] for _ in range(4)]
    assert indices == [1, 2, 3, 4] and source.ended
    assert FileFrameSource(clip, loop=True).ended is False


def test_capture_source_reads_a_network_stream(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    write_clip(clip, 5)
    capture = cv2.VideoCapture(str(clip))
    jpegs = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        jpegs.append(cv2.imencode(".jpg", frame)[1].tobytes())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                i = 0
                while True:
                    jpg = jpegs[i % len(jpegs)]
                    i += 1
                    header = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                    self.wfile.write(header + jpg + b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/video"
        try:
            source = CaptureFrameSource(SourceSpec("stream", url), open_timeout_s=5.0)
        except RuntimeError:
            pytest.skip("OpenCV's FFmpeg backend cannot open local HTTP streams here")
        assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ  # must not leak into later captures
        first = source.latest()[1]
        time.sleep(0.3)
        frame, later = source.latest()
        assert source.size == (64, 48) and later > first and frame.shape == (48, 64, 3)
        source.close()
    finally:
        server.shutdown()


def test_exponential_smoother() -> None:
    smoother = ExponentialSmoother(0.5)
    assert np.allclose(smoother.update(np.array([1.0, 2.0])), [1.0, 2.0])  # first sample passes through
    assert np.allclose(smoother.update(np.array([3.0, 2.0])), [2.0, 2.0])
    smoother.reset()
    assert np.allclose(smoother.update(np.array([9.0])), [9.0])
    with pytest.raises(ValueError):
        ExponentialSmoother(0.0)


def test_streaming_hysteresis_matches_batch() -> None:
    segmenter = GraspSegmenter(0.03, 0.05)
    aperture = np.array([0.07, 0.04, 0.02, 0.04, 0.06, 0.04, 0.02, 0.07])
    batch = segmenter.segment(aperture)
    state, streamed = False, []
    for value in aperture:
        state = segmenter.next_state(float(value), state)
        streamed.append(state)
    assert list(batch) == streamed == [False, False, True, True, False, False, True, False]


T_CAM_TABLE = make_transform(Rotation.from_euler("x", 180, degrees=True), [0.0, 0.0, 0.5])


class FakeMarkers:
    """Always sees the table marker at a fixed pose, with a little noise."""

    def estimate(self, image_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
        return {"table": T_CAM_TABLE}


class FakeHands:
    """Returns scripted (tcp in camera frame, aperture) pairs, then None (hand lost)."""

    def __init__(self, script: list[tuple[np.ndarray, float] | None]) -> None:
        self._script = list(script)

    def process_frame(self, image_bgr: np.ndarray, timestamp_ms: int) -> HandFrameResult | None:
        item = self._script.pop(0) if self._script else None
        if item is None:
            return None
        tcp_cam, aperture = item
        return HandFrameResult(np.eye(4), tcp_cam, aperture)


class FakeFrames:
    """A frame source that yields a new frame index on every query."""

    def __init__(self, num: int = 100) -> None:
        self._n, self._limit = 0, num

    @property
    def size(self) -> tuple[int, int]:
        return 64, 48

    @property
    def ended(self) -> bool:
        return self._n >= self._limit

    def latest(self) -> tuple[np.ndarray, int]:
        self._n += 1
        return np.zeros((48, 64, 3), np.uint8), self._n

    def close(self) -> None:
        pass


def build_live(script, calibration_frames: int = 2, keys=None, markers=True, **mapping_kwargs):
    tracker = LiveHandTracker(
        FakeHands(script),
        FakeMarkers() if markers else None,
        GraspSegmenter(0.03, 0.05),
        calibration_frames,
        smoothing_alpha=1.0,
    )
    anchors = []
    source = LiveHandCommandSource(
        FakeFrames(),
        tracker,
        make_mapping=lambda p_ref, p_world, t_ref_cam: TableToWorld(p_ref, p_world, **mapping_kwargs),
        keys=keys,
        on_anchor=lambda mapping, t_table_cam: anchors.append((mapping, t_table_cam)),
    )
    return source, anchors


def cam_point_over_table(x: float, y: float, height: float) -> np.ndarray:
    """Table-frame (x, y, height) as a camera-frame point: camera 0.5 m above the table, looking down."""
    p_table = np.array([x, y, height])
    return (T_CAM_TABLE @ np.append(p_table, 1.0))[:3]


def test_tracker_locks_table_then_reports_table_frame_tcp() -> None:
    script = [(cam_point_over_table(0.1, 0.0, 0.2), 0.06)] * 4
    tracker = LiveHandTracker(
        FakeHands(script), FakeMarkers(), GraspSegmenter(), calibration_frames=2, smoothing_alpha=1.0
    )
    frame = np.zeros((48, 64, 3), np.uint8)
    first = tracker.process(frame, 0.0)
    assert not first.table_ready and first.p_table_tcp is None  # hand seen, table not locked yet
    second = tracker.process(frame, 0.05)
    assert second.table_ready and np.allclose(second.p_table_tcp, [0.1, 0.0, 0.2], atol=1e-9)
    tracker.recalibrate()
    assert not tracker.table_ready


def test_live_source_holds_until_anchored_then_follows_deltas() -> None:
    eef = np.array([-0.2, 0.0, 1.1])
    script = [(cam_point_over_table(0.0, 0.0, 0.2), 0.06)] + [
        (cam_point_over_table(0.05 * k, 0.0, 0.2), 0.06) for k in range(1, 4)
    ]
    source, anchors = build_live(script, calibration_frames=1)
    commands = [source.next_command(eef) for _ in range(4)]
    assert anchors and anchors[0][0] is source.mapping
    assert commands[0].starts_stream and np.allclose(commands[0].pos_world, eef)  # first hand = robot pose
    assert not commands[1].starts_stream
    assert np.allclose(commands[3].pos_world, eef + [0.15, 0.0, 0.0], atol=1e-9)  # deltas


def test_live_source_holds_last_command_when_hand_is_lost_and_reports_it() -> None:
    eef = np.zeros(3)
    script = [(cam_point_over_table(0.0, 0.0, 0.2), 0.06), (cam_point_over_table(0.1, 0.0, 0.2), 0.06)]
    source, _ = build_live(script, calibration_frames=1)
    source.next_command(eef)
    moved = source.next_command(eef)
    lost = source.next_command(eef)  # fake hand tracker has run out: no hand
    assert np.allclose(lost.pos_world, moved.pos_world)
    assert source.status == "hand lost: holding"


def test_live_source_waits_for_table_and_reports_progress() -> None:
    source, _ = build_live([None] * 5, calibration_frames=3)
    command = source.next_command(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(command.pos_world, [1.0, 2.0, 3.0]) and not command.gripper_closed
    assert source.status.startswith("looking for the table marker (1/3)")


def test_live_source_reanchors_on_r_and_quits_on_escape() -> None:
    keys = ScriptedKeys(presses=[(2, "r"), (4, "escape")])
    script = [(cam_point_over_table(0.1, 0.0, 0.2), 0.06)] * 6
    source, anchors = build_live(script, calibration_frames=1, keys=keys)
    first_eef, moved_eef = np.array([0.0, 0.0, 1.0]), np.array([0.3, 0.0, 1.0])

    source.next_command(first_eef)
    assert len(anchors) == 1
    keys.step = 1
    assert not source.next_command(moved_eef).starts_stream  # same hand pose: just follows
    keys.step = 2
    again = source.next_command(moved_eef)  # 'r': the hand's current pose becomes the robot's current pose
    assert len(anchors) == 2 and again.starts_stream
    assert np.allclose(again.pos_world, moved_eef)
    keys.step = 4
    assert source.next_command(moved_eef) is None


def test_markerless_tracker_uses_camera_axes_immediately() -> None:
    hand_in_cam = np.array([0.05, -0.02, 0.4])
    tracker = LiveHandTracker(
        FakeHands([(hand_in_cam, 0.06)]), None, GraspSegmenter(), calibration_frames=10, smoothing_alpha=1.0
    )
    sample = tracker.process(np.zeros((48, 64, 3), np.uint8), 0.0)
    assert sample.table_ready  # nothing to wait for
    assert np.allclose(
        sample.p_table_tcp, hand_in_cam
    )  # camera-frame TCP, reported as the reference-frame TCP
    assert np.allclose(sample.t_cam_table, np.eye(4))


def test_level_camera_mapping_is_gravity_aligned_and_follows_agentview_heading() -> None:
    """Right -> +y, up -> +z, away from the camera -> -x (agentview looks along -x), whatever the pitch."""
    for pitch_deg in (0.0, 25.0):
        r_world_cam = level_camera_rotation(np.array([-1.0, 0.0]), np.radians(pitch_deg))
        assert np.allclose(r_world_cam.T @ r_world_cam, np.eye(3), atol=1e-12)
        assert np.linalg.det(r_world_cam) == pytest.approx(1.0)
        assert np.allclose(r_world_cam[:, 0], [0.0, 1.0, 0.0])  # image right is horizontal: +y
        assert r_world_cam[2, 1] < 0  # image down points to world down
    level = level_camera_rotation(np.array([-1.0, 0.0]))
    mapping = TableToWorld(np.zeros(3), np.array([-0.2, 0.0, 1.1]), rotation=level)
    origin = mapping.point_to_world(np.zeros(3))
    assert np.allclose(
        mapping.point_to_world(np.array([0.1, 0.0, 0.0])) - origin, [0.0, 0.1, 0.0], atol=1e-12
    )
    assert np.allclose(
        mapping.point_to_world(np.array([0.0, 0.1, 0.0])) - origin, [0.0, 0.0, -0.1], atol=1e-12
    )
    assert np.allclose(
        mapping.point_to_world(np.array([0.0, 0.0, 0.1])) - origin, [-0.1, 0.0, 0.0], atol=1e-12
    )
    pitched = level_camera_rotation(np.array([-1.0, 0.0]), np.radians(30.0))
    # Looking 30 deg down: the optical axis has a downward component, image-up is still not world-down.
    assert pitched[2, 2] == pytest.approx(-np.sin(np.radians(30.0)))


def test_markerless_live_source_follows_camera_frame_deltas() -> None:
    eef = np.array([-0.2, 0.0, 1.1])
    script = [(np.array([0.0, 0.0, 0.4]), 0.06), (np.array([0.1, 0.0, 0.4]), 0.06)]
    source, _ = build_live(script, markers=False)
    first, second = source.next_command(eef), source.next_command(eef)
    assert first.starts_stream and np.allclose(first.pos_world, eef)  # anchors on the very first hand
    assert np.allclose(second.pos_world, eef + [0.1, 0.0, 0.0])  # identity rotation here: camera x -> world x


def test_mirrored_source_flips_left_right_only() -> None:
    class One:
        size = (3, 1)
        ended = False

        def latest(self):
            return np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]], np.uint8), 7

        def close(self) -> None:
            pass

    frame, index = MirroredFrameSource(One()).latest()
    assert index == 7 and frame[0, :, 0].tolist() == [3, 2, 1]
