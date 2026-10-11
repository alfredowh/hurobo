"""Push the bowl around with a closed gripper to see how a collider behaves, then compare variants.

One run per collider (separate process: LIBERO's asset path is process-global):

    MUJOCO_GL=egl uv run python scripts/12_push_test.py --label stock
    MUJOCO_GL=egl uv run python scripts/12_push_test.py --label pink_bowl \\
        --collider-dir data/colliders/pink_bowl
    MUJOCO_GL=egl uv run python scripts/12_push_test.py --label fruit_bowl \\
        --collider-dir data/colliders/fruit_bowl
    uv run python scripts/12_push_test.py --combine stock pink_bowl fruit_bowl

Each frame: LIBERO's own render (left) | the splat render with the bowl's physics
shape tinted red (right). The pusher is scripted and identical in every run; the
only thing that changes is the bowl's collision geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.collider import use_assets_dir
from hurobo.control import Waypoint, WaypointFollower
from hurobo.sim import LiberoTask, ObjectPose
from hurobo.splat import (
    LIBERO_GOAL_DISTRACTOR_ROOTS,
    SplatCompositor,
    SplatFit,
    build_assets,
)
from hurobo.utils.plotting import plot_push_traces
from hurobo.utils.video import VideoWriter, caption_frame, read_frames

BOWL_JOINT = "akita_black_bowl_1_joint0"
BOWL_BODY = "akita_black_bowl_1_main"
#: Pusher fingertip height above the bowl's resting origin: mid-wall of the stock bowl.
PUSH_HEIGHT_M = 0.03
HOVER_HEIGHT_M = 0.12
#: Stand-off from the bowl centre at the start of a push: bowl radius (~0.08) + finger clearance.
START_STANDOFF_M = 0.125
GRIPPER_CLOSE_STEPS = 15
#: Scales the follower's [-1, 1] output: its clip bound doubles as its normaliser, so at full
#: scale it commands the controller's maximum 5 cm/step, which destabilised the arm (checked).
PUSH_ACTION_SCALE = 0.3
#: Loose scene objects that stand in the pusher's way; they are not drawn in the splat render anyway.
PARKED_OBJECTS = {"wine_bottle_1_joint0": (0.30, 0.42), "cream_cheese_1_joint0": (0.30, -0.42)}
#: Fixtures that box in the pusher (the cabinet blocked the -y approach, checked); moved below the floor.
REMOVED_FIXTURE_ROOTS = ("wooden_cabinet_1_main", "flat_stove_1_main", "wine_rack_1_main")
FIXTURE_DROP_M = 3.0
MAX_STEPS_PER_PUSH = 140
#: (direction in world x-y, distance in metres), each applied to the bowl's pose at that moment.
PUSHES = [
    (np.array([0.0, 1.0]), 0.12),
    (np.array([-1.0, 0.0]), 0.10),
    (np.array([1.0, 1.0]) / np.sqrt(2), 0.12),
    (np.array([0.0, -1.0]), 0.16),
]


def push_waypoints(
    p_world_bowl: np.ndarray, p_world_eef: np.ndarray, eef_rot, direction_xy: np.ndarray, distance_m: float
) -> list[Waypoint]:
    """Waypoints for one push: lift, hover behind the bowl, descend, sweep through it, lift."""
    z_push = p_world_bowl[2] + PUSH_HEIGHT_M
    z_hover = p_world_bowl[2] + HOVER_HEIGHT_M
    start = p_world_bowl[:2] - direction_xy * START_STANDOFF_M
    end = p_world_bowl[:2] + direction_xy * distance_m
    xyz = lambda xy, z: np.array([xy[0], xy[1], z])  # noqa: E731
    points = [
        np.array([p_world_eef[0], p_world_eef[1], z_hover]),
        xyz(start, z_hover),
        xyz(start, z_push),
        xyz(end, z_push),
        xyz(end, z_hover),
    ]
    return [Waypoint(p, eef_rot, gripper_closed=True) for p in points]


def tilt_degrees(task: LiberoTask) -> float:
    """Angle between the bowl's axis and world up."""
    axis_world = task.get_object_pose(BOWL_JOINT).rot.apply([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(axis_world[2], -1.0, 1.0))))


def run_push_test(args: argparse.Namespace) -> None:
    """Run the scripted pushes against one collider variant; write video, trace and summary."""
    out = args.out / args.label
    out.mkdir(parents=True, exist_ok=True)

    bowl_ply, bowl_fit, bowl_yaw, title = args.bowl_ply, SplatFit.MATCH_BODY_XY, 0.0, "stock LIBERO collider"
    if args.collider_dir:
        meta = json.loads((args.collider_dir / "collider.json").read_text())
        use_assets_dir(Path(meta["assets_dir"]))
        bowl_ply, bowl_fit, bowl_yaw = Path(meta["ply"]), SplatFit(meta["fit"]), meta["yaw_deg"]
        title = f"scan collider: {meta['name']} ({meta['num_pieces']} convex pieces)"

    task = LiberoTask(args.suite, args.task_id)
    obs = task.reset()
    for joint_name, (x, y) in PARKED_OBJECTS.items():
        z = task.get_object_pose(joint_name).pos[2]
        task.set_object_pose(joint_name, ObjectPose(np.array([x, y, z]), Rotation.identity()))
    model, data = task.sim.model._model, task.sim.data._data
    for root_name in REMOVED_FIXTURE_ROOTS:
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        if model.body_jntnum[root] != 0:
            raise RuntimeError(f"{root_name} has joints; this only moves welded fixtures")
        model.body_pos[root, 2] -= FIXTURE_DROP_M
    task.sim.forward()
    compositor = SplatCompositor(
        model,
        build_assets(bowl_ply, None, None, bowl_fit, bowl_yaw),
        hidden_body_roots=LIBERO_GOAL_DISTRACTOR_ROOTS,
        width=args.size,
        height=args.size,
    )
    pos_scale_m, rot_scale_rad = task.controller_action_scale
    # Coarser than the grasp demo's 5 mm: tiny commanded deltas don't move the OSC arm (checked), and
    # a push doesn't need millimetre waypoint accuracy.
    follower = WaypointFollower(
        max_pos_delta_m=pos_scale_m,
        max_rot_delta_rad=rot_scale_rad,
        pos_reach_threshold_m=0.015,
        rot_reach_threshold_rad=0.2,
    )
    _, eef_rot = task.get_eef_pose(obs)

    positions, tilts, eef_positions = [], [], []

    def record(writer: VideoWriter) -> None:
        pose = task.get_object_pose(BOWL_JOINT)
        positions.append(pose.pos.copy())
        tilts.append(tilt_degrees(task))
        eef_positions.append(task.get_eef_pose(obs)[0].copy())
        native = compositor.render_native(data, args.camera)
        splat = compositor.render(data, args.camera).composite.copy()
        mask = compositor.render_collision_mask(data, args.camera, BOWL_BODY)
        splat[mask] = (0.45 * splat[mask] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)
        left = caption_frame(native, "LIBERO render")
        right = caption_frame(splat, "splat + physics shape (red)")
        panel = np.concatenate([left, right], axis=1)
        cv2.putText(
            panel, title, (10, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            panel,
            title,
            (10, panel.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(panel)

    closed = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
    with VideoWriter(out / "push.mp4", fps=20.0, frame_size=(2 * args.size, args.size)) as writer:
        record(writer)
        for _ in range(GRIPPER_CLOSE_STEPS):
            obs, _, _ = task.step(closed)
            record(writer)
        for direction, distance in PUSHES:
            p_bowl = task.get_object_pose(BOWL_JOINT).pos
            follower.set_waypoints(
                push_waypoints(p_bowl, task.get_eef_pose(obs)[0], eef_rot, direction, distance)
            )
            for _ in range(MAX_STEPS_PER_PUSH):
                if follower.done:
                    break
                pos, rot = task.get_eef_pose(obs)
                action = follower.step(pos, rot)
                action[:6] *= PUSH_ACTION_SCALE
                obs, _, _ = task.step(action)
                record(writer)
        for _ in range(20):  # let it come to rest, to see whether anything keeps drifting
            obs, _, _ = task.step(closed)
            record(writer)

    pos = np.array(positions)
    tilt = np.array(tilts)
    np.savez(out / "trace.npz", pos=pos, tilt_deg=tilt, eef=np.array(eef_positions))
    summary = {
        "label": args.label,
        "title": title,
        "frames": len(pos),
        "bowl_path_length_m": float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))),
        "bowl_net_displacement_m": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
        "max_tilt_deg": float(tilt.max()),
        "height_change_mm_min_max": [
            float(1000 * (pos[:, 2] - pos[0, 2]).min()),
            float(1000 * (pos[:, 2] - pos[0, 2]).max()),
        ],
        "final_speed_mm_per_step": float(1000 * np.linalg.norm(pos[-1] - pos[-2])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    compositor.close()
    task.close()


def combine(args: argparse.Namespace) -> None:
    """Stack the runs' videos row by row (shorter ones hold their last frame) and plot their traces."""
    runs = [args.out / label for label in args.combine]
    clips = [list(read_frames(run / "push.mp4")) for run in runs]
    length = max(len(c) for c in clips)
    height, width = clips[0][0].shape[:2]
    with VideoWriter(
        args.out / "push_comparison.mp4", fps=20.0, frame_size=(width, height * len(runs))
    ) as writer:
        for i in range(length):
            writer.write(np.concatenate([c[min(i, len(c) - 1)] for c in clips], axis=0))
    traces = {}
    for label, run in zip(args.combine, runs, strict=True):
        data = np.load(run / "trace.npz")
        traces[label] = {"pos": data["pos"], "tilt_deg": data["tilt_deg"]}
    plot_push_traces(traces, args.out / "push_traces.png")
    print(f"wrote {args.out / 'push_comparison.mp4'} and {args.out / 'push_traces.png'}")


def main() -> None:
    """Parse args; either run one push test or combine finished runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", help="run name; output goes to <out>/<label>/")
    parser.add_argument(
        "--collider-dir", type=Path, help="10_build_collider.py output; default: stock collider"
    )
    parser.add_argument(
        "--bowl-ply", type=Path, help="splat for the stock run (default: DISCOVERSE pink bowl)"
    )
    parser.add_argument("--combine", nargs="+", help="labels of finished runs to stack into one video + plot")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--out", type=Path, default=Path("data/push_test"))
    args = parser.parse_args()
    if args.combine:
        combine(args)
    elif args.label:
        run_push_test(args)
    else:
        parser.error("give --label (run one test) or --combine (compare finished runs)")


if __name__ == "__main__":
    main()
