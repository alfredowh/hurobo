#!/usr/bin/env python
"""Drives the LIBERO Panda from a human hand clip or from the keyboard, with a live view and videos.

Hand clip (stand-in markers pasted into a frame folder), recorded and watched live:

    MUJOCO_GL=egl uv run python scripts/13_teleop.py hand data/raw/overview_gif/hand_moves_cube --live

Live camera (with --no-markers only the relative hand motion is used; otherwise the printed table
marker must be in view; see docs/camera_sources.md):

    uv run python scripts/13_teleop.py sources            # what cameras this machine has
    MUJOCO_GL=egl uv run python scripts/13_teleop.py hand webcam:0 --no-markers   # no printed marker needed
    MUJOCO_GL=egl uv run python scripts/13_teleop.py hand webcam:0                 # with the table marker
    MUJOCO_GL=egl uv run python scripts/13_teleop.py hand /dev/video4       # e.g. phone via scrcpy
    MUJOCO_GL=egl uv run python scripts/13_teleop.py hand http://PHONE_IP:8080/video

Keyboard (arrows move in the agentview, w/s up/down, space toggles the gripper, esc quits):

    MUJOCO_GL=egl uv run python scripts/13_teleop.py keyboard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.camera_source import (
    MirroredFrameSource,
    SourceSpec,
    list_cameras,
    open_frame_source,
    parse_source,
)
from hurobo.live import LiveWindow
from hurobo.markers import MarkerLayout
from hurobo.props import CubeProp
from hurobo.sim import LiberoTask
from hurobo.teleop import KeyboardCommandSource, TeleopSession, WorkspaceLimits
from hurobo.teleop_app import (
    PreparedClip,
    TeleopFrameComposer,
    TeleopRunner,
    default_layout,
    guess_intrinsics,
    prepare_stand_in_clip,
    prepare_video_clip,
    reset_with_props,
    run_hand_teleop,
    run_live_teleop,
    write_run_videos,
    write_standin_slow_video,
    write_trace_video,
)


def add_common_options(parser: argparse.ArgumentParser, suppress_defaults: bool = False) -> None:
    """Options shared by both modes; accepted before or after the mode name.

    With `suppress_defaults` (the copy on the sub-commands) an option left out does not overwrite
    the value given before the mode name.
    """
    default = (lambda value: argparse.SUPPRESS) if suppress_defaults else (lambda value: value)
    parser.add_argument("--suite", default=default("libero_goal"))
    parser.add_argument("--task-id", type=int, default=default(8))
    parser.add_argument(
        "--no-cube", action="store_true", default=default(False), help="do not spawn the cube on grasp"
    )
    parser.add_argument("--panel-height", type=int, default=default(300))
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=default(False),
        help="open the live window full screen (F11 toggles); the view is scaled to fit",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=default(False),
        help="show a live window (keyboard mode always does)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_options(parser)
    modes = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    add_common_options(common, suppress_defaults=True)

    modes.add_parser("sources", help="list the cameras on this machine and how to add a phone")

    hand = modes.add_parser(
        "hand", parents=[common], help="drive the robot from a hand clip or a live camera"
    )
    hand.add_argument(
        "source",
        help="image folder or video file (offline replay); webcam[:N], N, /dev/videoN, http(s)/rtsp "
        "URL, or file:<video> (live camera pipeline)",
    )
    hand.add_argument("--out", type=Path, default=None, help="default: data/teleop/<source name>")
    hand.add_argument(
        "--resolution", default="1280x720", help="requested webcam frame size WxH (live sources)"
    )
    hand.add_argument("--fps", type=float, default=30.0, help="requested webcam frame rate (live sources)")
    hand.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="stop a live run after this long (default: until quit)",
    )
    hand.add_argument(
        "--no-markers",
        action="store_true",
        help="live sources: no table marker, relative hand motion only (camera is the reference frame, "
        "assumed level and facing the way the agentview camera does)",
    )
    hand.add_argument(
        "--camera-pitch",
        type=float,
        default=0.0,
        help="with --no-markers: degrees the camera looks below horizontal (tilted-back laptop: negative)",
    )
    hand.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="flip the live image left-right like a mirror (default: on for webcams with --no-markers; "
        "never with markers, since ArUco cannot read mirrored markers)",
    )
    hand.add_argument("--calib-frames", type=int, default=10, help="table-marker detections to average")
    hand.add_argument(
        "--headless", action="store_true", help="live source without a window (needs --max-seconds)"
    )
    hand.add_argument(
        "--record", action="store_true", help="live source: also write data/teleop/.../teleop_live.mp4"
    )
    hand.add_argument(
        "--slowdown", type=float, default=4.0, help="play the human motion this many times slower"
    )
    hand.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=(0.0, -0.45, 0.40),
        metavar=("X", "Y", "Z"),
        help="stand-in camera position in the table frame (image folders only)",
    )
    hand.add_argument(
        "--align",
        default="agentview",
        help="'agentview' (match its heading), 'table' (no rotation) or a yaw in degrees",
    )
    hand.add_argument("--scale", type=float, default=1.0, help="robot metres per hand metre")
    hand.add_argument(
        "--table-height",
        action="store_true",
        help="map the hand's height above the real table to height above the sim table "
        "(robot first moves there); default: pure deltas from the robot's start pose",
    )
    hand.add_argument(
        "--marker-height-m",
        type=float,
        default=0.0,
        help="height of the marker plane above the real table top, e.g. 0.0083 for a Surface Go lying "
        "on the table showing the marker; only matters with --table-height",
    )
    hand.add_argument("--no-record", action="store_true", help="skip the mp4 outputs")
    hand.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("data/camera_intrinsics.npz"),
        help="from scripts/01_calibrate_camera.py; live sources fall back to a guess if it is missing",
    )
    hand.add_argument("--table-id", type=int, default=20)
    hand.add_argument("--bowl-id", type=int, default=21)
    hand.add_argument("--plate-id", type=int, default=22)
    hand.add_argument("--table-size-m", type=float, default=0.10)
    hand.add_argument("--object-size-m", type=float, default=0.05)

    keyboard = modes.add_parser(
        "keyboard", parents=[common], help="drive the robot and gripper with the keyboard (always live)"
    )
    keyboard.add_argument("--speed", type=float, default=0.15, help="target speed in m/s while a key is held")
    return parser.parse_args()


def marker_layout(args: argparse.Namespace) -> MarkerLayout:
    return MarkerLayout(
        args.table_id,
        args.table_size_m,
        {"bowl": args.bowl_id, "plate": args.plate_id},
        args.object_size_m,
    )


def load_clip(spec: SourceSpec, args: argparse.Namespace, out_dir: Path) -> PreparedClip:
    """Stand-in extraction for a frame folder, real extraction for a video file."""
    if spec.kind == "frames":
        return prepare_stand_in_clip(Path(str(spec.target)), out_dir, tuple(args.camera_pos))
    return prepare_video_clip(
        Path(str(spec.target)), CameraIntrinsics.load(args.intrinsics), marker_layout(args)
    )


def run_hand(args: argparse.Namespace, task: LiberoTask, cube: CubeProp | None) -> None:
    spec = parse_source(args.source)
    if spec.is_live:
        run_live(spec, args, task, cube)
        return
    out_dir = args.out or Path("data/teleop") / Path(args.source).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = load_clip(spec, args, out_dir)
    demo = clip.demo
    print(
        f"clip: {len(demo.timestamps_s)} samples over {demo.timestamps_s[-1]:.2f}s, "
        f"grasp@{demo.grasp_frame_idx} release@{demo.release_frame_idx}, "
        f"QC: {'; '.join(clip.report.reasons) or 'all gates passed'}"
    )

    window = LiveWindow("hurobo: hand teleop", args.fullscreen) if args.live else None
    result = run_hand_teleop(
        task,
        cube,
        clip,
        args.slowdown,
        args.align,
        args.scale,
        args.table_height,
        args.marker_height_m,
        args.panel_height,
        record=not args.no_record,
        window=window,
    )
    if window is not None:
        window.close()
    trace = result.trace
    print(
        f"table->world yaw {np.degrees(result.yaw_rad):.1f} deg, scale {args.scale}, "
        f"anchor z {result.mapping.p_world_anchor[2]:.3f}"
    )

    error_cm = 100 * trace.tracking_error_m()
    print(
        f"{len(trace.t_s)} steps; tracking error mean {error_cm.mean():.1f} cm, max {error_cm.max():.1f} cm; "
        f"cube spawned at step {trace.cube_spawn_step}"
    )
    if not args.no_record:
        paths = write_run_videos(trace, out_dir, args.slowdown)
        paths["human_slow"] = write_standin_slow_video(clip, out_dir, args.slowdown)
        paths["human_1x"] = clip.video_path
        print("videos:", *(f"\n  {name}: {path}" for name, path in paths.items()))


def run_live(spec: SourceSpec, args: argparse.Namespace, task: LiberoTask, cube: CubeProp | None) -> None:
    """Drive the arm from a camera (webcam, V4L2 phone bridge, network stream, or a file played as one)."""
    if args.headless and args.max_seconds is None:
        raise SystemExit("--headless needs --max-seconds, otherwise nothing can end the run")
    width, height = (int(v) for v in args.resolution.lower().split("x"))
    if args.no_markers and args.table_height:
        raise SystemExit("--table-height needs the table marker; drop it or drop --no-markers")
    frames = open_frame_source(spec, width, height, args.fps)
    use_markers = not args.no_markers
    if args.mirror and use_markers:
        frames.close()
        raise SystemExit("--mirror makes the table marker unreadable; drop --mirror or add --no-markers")
    if args.mirror if args.mirror is not None else (spec.kind == "webcam" and not use_markers):
        frames = MirroredFrameSource(frames)
    if args.intrinsics.exists():
        intrinsics = CameraIntrinsics.load(args.intrinsics)
        if (intrinsics.image_width, intrinsics.image_height) != frames.size:
            frames.close()
            raise SystemExit(
                f"{args.intrinsics} was calibrated at {intrinsics.image_width}x{intrinsics.image_height} "
                f"but the camera delivers {frames.size[0]}x{frames.size[1]}; pass --resolution or recalibrate"
            )
    else:
        intrinsics = guess_intrinsics(frames.size)
        print(
            f"WARNING: {args.intrinsics} not found; guessing intrinsics (fx = {intrinsics.fx:.0f} px). "
            "Hand scale and depth are only roughly right; run scripts/01_calibrate_camera.py capture."
        )
    print(f"camera {spec.target!r}: {frames.size[0]}x{frames.size[1]}")
    window = None if args.headless else LiveWindow("hurobo: live hand teleop", args.fullscreen)
    max_steps = None if args.max_seconds is None else int(args.max_seconds * 20)
    result = run_live_teleop(
        task,
        cube,
        frames,
        intrinsics,
        None if args.no_markers else marker_layout(args),
        args.align,
        args.scale,
        args.table_height,
        args.marker_height_m,
        args.panel_height,
        record=args.record,
        window=window,
        max_steps=max_steps,
        calibration_frames=args.calib_frames,
        camera_pitch_deg=args.camera_pitch,
        on_event=lambda line: print(f"[live] {line}", flush=True),
    )
    if window is not None:
        window.close()
    trace = result.trace
    print(f"{len(trace.t_s)} steps; final status: {result.source.status}; cube step {trace.cube_spawn_step}")
    if args.record and trace.frames:
        out_dir = args.out or Path("data/teleop/live")
        out_dir.mkdir(parents=True, exist_ok=True)
        print("video:", write_trace_video(trace, out_dir / "teleop_live.mp4"))


def list_sources() -> None:
    """Print the V4L2 nodes the kernel knows about, and the ways to add a phone."""
    cameras = list_cameras()
    print("V4L2 video nodes (OpenCV index = the number in /dev/videoN):")
    if not cameras:
        print("  none found")
    for camera in cameras:
        note = "" if camera.is_capture_node else "   (metadata node, not a picture source)"
        print(f"  {camera.device:14s} {camera.name}{note}")
    print(
        "\nUse one as:  hand webcam:N   or   hand /dev/videoN\n"
        "A phone shows up here once a bridge runs (scrcpy + v4l2loopback, DroidCam, or USB webcam mode),\n"
        "or needs no device at all as a URL:  hand http://PHONE_IP:8080/video\n"
        "Setup steps for each: docs/camera_sources.md"
    )


def run_keyboard(args: argparse.Namespace, task: LiberoTask, cube: CubeProp | None) -> None:
    obs = reset_with_props(task, cube)
    p_world_eef0, _ = task.get_eef_pose(obs)
    window = LiveWindow("hurobo: keyboard teleop", args.fullscreen)
    source = KeyboardCommandSource(window, p_world_eef0, speed_mps=args.speed)
    session = TeleopSession(task, source, WorkspaceLimits.around(p_world_eef0), cube)
    composer = TeleopFrameComposer(task, default_layout(False, task.TWIN_CAMERA_NAME), args.panel_height)
    TeleopRunner(session, composer, window).run(obs)
    window.close()


def main() -> None:
    args = parse_args()
    if args.mode == "sources":
        list_sources()
        return
    task = LiberoTask(args.suite, args.task_id)
    cube = None if args.no_cube else CubeProp()
    try:
        (run_hand if args.mode == "hand" else run_keyboard)(args, task, cube)
    finally:
        task.close()


if __name__ == "__main__":
    main()
