"""Photoreal real2sim demo: a LIBERO rollout rendered with real-object splats, beside native LIBERO.

Stretch goal (PLAN.md §5, REAL2SIM_IDEAS.md §7). Until the S24 orbit of the real
table is captured and splatted, the defaults stand in with DISCOVERSE's
real-scanned splats (pink bowl, white plate, table cloth; HF `tatp/DISCOVERSE-models`).
The robot is always LIBERO's mesh Panda, never a scan. The motion is a scripted
rim grasp, only there to show splats following bodies — not a policy result.

    MUJOCO_GL=egl uv run python scripts/08_splat_composite.py [--bowl-ply data/splats/my_bowl.ply]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from hurobo.collider import use_assets_dir
from hurobo.control import Waypoint, WaypointFollower
from hurobo.sim import LiberoTask
from hurobo.splat import (
    LIBERO_GOAL_DISTRACTOR_ROOTS,
    CompositeFrame,
    SplatCompositor,
    SplatFit,
    build_assets,
)
from hurobo.utils.video import VideoWriter
from hurobo.utils.video import caption_frame as label

#: Rim grasp of the akita bowl: fingers open along world y, so grasp the wall on
#: the +y side where it runs along x. Values read off the bowl's visual AABB
#: (radius ~0.078 m, height ~0.053 m), not tuned on outcomes.
BOWL_RIM_OFFSET_P_BOWL = np.array([0.0, 0.068, 0.035])
APPROACH_HEIGHT_M = 0.12
GRIPPER_SETTLE_STEPS = 15


def pick_place_waypoints(task: LiberoTask, eef_rot, grasp_offset_p_bowl: np.ndarray) -> list[Waypoint]:
    """Scripted bowl-onto-plate waypoints from the sim's current object poses (world frame)."""
    p_world_bowl = task.get_object_pose("akita_black_bowl_1_joint0").pos
    p_world_plate = task.get_object_pose("plate_1_joint0").pos
    p_world_grasp = p_world_bowl + grasp_offset_p_bowl
    p_world_release = p_world_plate + grasp_offset_p_bowl + np.array([0.0, 0.0, 0.03])
    up = np.array([0.0, 0.0, APPROACH_HEIGHT_M])
    return [
        Waypoint(p_world_grasp + up, eef_rot, gripper_closed=False),
        Waypoint(p_world_grasp, eef_rot, gripper_closed=False),
        Waypoint(p_world_grasp, eef_rot, gripper_closed=True),
        Waypoint(p_world_grasp + up, eef_rot, gripper_closed=True),
        Waypoint(p_world_release + up, eef_rot, gripper_closed=True),
        Waypoint(p_world_release, eef_rot, gripper_closed=True),
        Waypoint(p_world_release, eef_rot, gripper_closed=False),
        Waypoint(p_world_release + up, eef_rot, gripper_closed=False),
    ]


def layers_figure(native: np.ndarray, frame: CompositeFrame) -> np.ndarray:
    """One-row BGR figure: native | mesh layer | splat layer | composite."""
    return np.concatenate(
        [
            label(native, "LIBERO native"),
            label(frame.mesh_rgb, "mesh layer (robot)"),
            label(frame.splat_rgb, "splat layer"),
            label(frame.composite, "depth composite"),
        ],
        axis=1,
    )


def main() -> None:
    """Parse args, roll out the scripted motion, write the comparison video and figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bowl-ply", type=Path, help="splat for the bowl (default: DISCOVERSE pink bowl)")
    parser.add_argument(
        "--plate-ply", type=Path, help="splat for the plate (default: DISCOVERSE white plate)"
    )
    parser.add_argument(
        "--table-ply", type=Path, help="splat for the table (default: DISCOVERSE table cloth)"
    )
    parser.add_argument(
        "--collider-dir",
        type=Path,
        help="output of 10_build_collider.py: run the sim on the scan-derived bowl collider and its splat",
    )
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("data/splat_demo"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grasp_offset = BOWL_RIM_OFFSET_P_BOWL
    bowl_ply, bowl_fit, bowl_yaw = args.bowl_ply, SplatFit.MATCH_BODY_XY, 0.0
    if args.collider_dir:
        meta = json.loads((args.collider_dir / "collider.json").read_text())
        use_assets_dir(Path(meta["assets_dir"]))
        grasp_offset = np.array(meta["grasp_offset_p_bowl"])
        bowl_ply, bowl_fit, bowl_yaw = Path(meta["ply"]), SplatFit(meta["fit"]), meta["yaw_deg"]
        print(f"collider: {meta['num_pieces']} convex pieces replace {meta['replaced_geoms']} stock boxes")

    task = LiberoTask(args.suite, args.task_id)
    obs = task.reset()
    model, data = task.mj_model, task.mj_data
    compositor = SplatCompositor(
        model,
        build_assets(bowl_ply, args.plate_ply, args.table_ply, bowl_fit, bowl_yaw),
        hidden_body_roots=LIBERO_GOAL_DISTRACTOR_ROOTS,
        width=args.size,
        height=args.size,
    )
    print(f"{compositor.num_gaussians} Gaussians attached")

    _, eef_rot = task.get_eef_pose(obs)
    pos_scale_m, rot_scale_rad = task.controller_action_scale
    # Normalise by the controller's real output scale so a unit action means what it says.
    follower = WaypointFollower(max_pos_delta_m=pos_scale_m, max_rot_delta_rad=rot_scale_rad)
    follower.set_waypoints(pick_place_waypoints(task, eef_rot, grasp_offset))

    # Settle test: with the arm idle, a sane collider leaves the bowl where it was spawned.
    z_before = task.get_object_pose("akita_black_bowl_1_joint0").pos[2]
    for _ in range(20):
        obs, _, _ = task.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
    z_drift_mm = 1000 * (task.get_object_pose("akita_black_bowl_1_joint0").pos[2] - z_before)
    print(f"settle test: bowl z moved {z_drift_mm:+.1f} mm over 20 idle steps")

    render_seconds, success, settle_left, last_gripper = [], False, 0, -1.0
    with VideoWriter(
        args.out / "native_vs_splat.mp4", fps=20.0, frame_size=(2 * args.size, args.size)
    ) as video:
        for step in range(args.max_steps):
            pos, rot = task.get_eef_pose(obs)
            action = follower.step(pos, rot)
            # Hold still while the fingers close/open; the follower alone would move on at once.
            if action[6] != last_gripper:
                settle_left, last_gripper = GRIPPER_SETTLE_STEPS, action[6]
            if settle_left > 0:
                action[:6] = 0.0
                settle_left -= 1
            obs, success, terminated = task.step(action)

            t0 = time.perf_counter()
            frame = compositor.render(data, args.camera)
            render_seconds.append(time.perf_counter() - t0)
            native = compositor.render_native(data, args.camera)
            video.write(
                np.concatenate([label(native, "LIBERO"), label(frame.composite, "splat twin")], axis=1)
            )
            if step % 40 == 0:
                cv2.imwrite(str(args.out / f"layers_{step:03d}.png"), layers_figure(native, frame))
                mask = compositor.render_collision_mask(data, args.camera, "akita_black_bowl_1_main")
                overlay = frame.composite.copy()
                overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)
                cv2.imwrite(
                    str(args.out / f"collider_overlay_{step:03d}.png"), label(overlay, "collider (red)")
                )
            if (follower.done and settle_left == 0) or terminated:
                break

    cv2.imwrite(str(args.out / "layers_final.png"), layers_figure(native, frame))
    compositor.close()
    task.close()
    print(
        f"steps={step + 1} task_success={success} "
        f"composite render {1000 * np.median(render_seconds):.1f} ms/frame (median) -> {args.out}"
    )


if __name__ == "__main__":
    main()
