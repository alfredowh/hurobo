"""Planning a recording session: where the phone goes, and which layouts and hand starts to record."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from hurobo.camera import CameraIntrinsics
from hurobo.real_setup import RealSetup
from hurobo.scaled_native import LayoutSampler, SimLayout, Workspace, table_to_world_map

#: How far the hand reaches above the tabletop while moving the bowl.
HAND_ENVELOPE_HEIGHT_M = 0.30


@dataclass(frozen=True)
class CameraPlacement:
    """A phone pose on the tripod and what it sees.

    Attributes:
        position_table: shape (3,), the lens position in the table frame.
        aim_table: shape (3,), the point the optical axis passes through.
        t_table_cam: 4x4 pose of the camera in the table frame (OpenCV axes: x right, y down, z forward).
        min_margin: smallest distance of any required point from the image border, as a fraction of the
            image size (0.05 = 5 %).
        board_width_px: width of the ChArUco grid in pixels.
        hand_width_px: width a 9 cm hand has at the workspace centre.
        elevation_deg: angle of the optical axis below the horizon.
        height_m: lens height above the tabletop.
        horizontal_fov_deg: horizontal field of view of the camera.
    """

    position_table: np.ndarray
    aim_table: np.ndarray
    t_table_cam: np.ndarray
    min_margin: float
    board_width_px: float
    hand_width_px: float
    elevation_deg: float
    height_m: float
    horizontal_fov_deg: float = 71.0


def camera_pose_looking_at(position: np.ndarray, aim: np.ndarray) -> np.ndarray:
    """T_table_cam (OpenCV axes) for a level camera at `position` looking at `aim`, table frame z up.

    Args:
        position: shape (3,).
        aim: shape (3,), not directly above or below `position`.

    Returns:
        4x4 transform.
    """
    forward = np.asarray(aim, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    return pose


def project_points(
    points_table: np.ndarray, t_table_cam: np.ndarray, intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels and depths of table-frame points seen from a camera pose (no lens distortion).

    Args:
        points_table: shape (N, 3).
        t_table_cam: 4x4 camera pose.
        intrinsics: the camera.

    Returns:
        (pixels (N, 2), depth_m (N,)); pixels of points behind the camera are meaningless.
    """
    rotation, position = t_table_cam[:3, :3], t_table_cam[:3, 3]
    points_cam = (np.asarray(points_table, float) - position) @ rotation
    depth = points_cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    pixels = np.column_stack(
        [
            intrinsics.fx * points_cam[:, 0] / safe + intrinsics.camera_matrix[0, 2],
            intrinsics.fy * points_cam[:, 1] / safe + intrinsics.camera_matrix[1, 2],
        ]
    )
    return pixels, depth


class CameraPlacementSearch:
    """Finds phone poses from which the board, both objects and the hand's whole reach are in view.

    The score is the margin to the image border of the worst required point, so the pose with the most
    slack wins. Hard limits (board large enough and not at a grazing angle, hand large enough, lens above
    the table, tripod not on the table or in the person's seat) discard the rest.
    """

    def __init__(
        self,
        setup: RealSetup,
        workspace: Workspace,
        intrinsics: CameraIntrinsics,
        min_board_width_px: float = 150.0,
        min_hand_width_px: float = 90.0,
        max_board_incidence_deg: float = 65.0,
        min_elevation_deg: float = 35.0,
    ) -> None:
        """Set the scene and the limits.

        Args:
            setup: measured table, board and objects.
            workspace: world rectangle of object centres (mapped to the table by `table_to_world_map`).
            intrinsics: the phone camera at its recording resolution.
            min_board_width_px: board grid width needed for reliable corner detection.
            min_hand_width_px: a 9 cm hand must be at least this wide at the workspace centre.
            max_board_incidence_deg: largest angle between the view ray and the board normal.
            min_elevation_deg: lowest angle of the line of sight to the work-area centre above the
                horizon. A flat view makes the height of the hand ambiguous.
        """
        self._setup = setup
        self._intrinsics = intrinsics
        self._min_board_width_px = min_board_width_px
        self._min_hand_width_px = min_hand_width_px
        self._max_incidence_rad = np.radians(max_board_incidence_deg)
        self._min_elevation_rad = np.radians(min_elevation_deg)
        self._mapping = table_to_world_map(setup, workspace)
        corners_world = np.array(
            [[workspace.x_range[i], workspace.y_range[j], 0.9] for i in (0, 1) for j in (0, 1)]
        )
        self._work_corners_table = np.array([self._mapping.point_to_table(p) for p in corners_world])
        self._work_centre_table = np.array(
            [*self._mapping.point_to_table(np.array([*workspace.centre, 0.9]))[:2], 0.0]
        )
        # Footprints of the objects at the workspace edge, and the hand's reach above them.
        margin = max(setup.plate.radius_m, 0.10)
        low = self._work_corners_table[:, :2].min(axis=0) - margin
        high = self._work_corners_table[:, :2].max(axis=0) + margin
        xs, ys = np.linspace(low[0], high[0], 5), np.linspace(low[1], high[1], 5)
        floor = np.array([[x, y, 0.0] for x in xs for y in ys])
        reach = floor.copy()
        reach[:, 2] = HAND_ENVELOPE_HEIGHT_M
        self._required = np.vstack([floor, reach])
        board = setup.board
        self._board_points = np.array(
            [
                [0.0, 0.0, board.thickness_m],
                [board.width_m, 0.0, board.thickness_m],
                [board.width_m, -board.height_m, board.thickness_m],
                [0.0, -board.height_m, board.thickness_m],
            ]
        )
        self._board_centre = self._board_points.mean(axis=0)
        table = setup.table
        self._table_box = (
            table.left_edge_x_m - 0.10,
            table.right_edge_x_m + 0.10,
            table.near_edge_y_m - 0.10,
            table.far_edge_y_m + 0.10,
        )

    @property
    def workspace_centre_table(self) -> np.ndarray:
        """shape (3,), the middle of the work area on the tabletop."""
        return self._work_centre_table

    def evaluate(self, position_table: np.ndarray, aim_table: np.ndarray) -> CameraPlacement | None:
        """Score one pose; None if a hard limit is violated.

        Args:
            position_table: shape (3,).
            aim_table: shape (3,).
        """
        position = np.asarray(position_table, float)
        x_low, x_high, y_low, y_high = self._table_box
        if x_low < position[0] < x_high and y_low < position[1] < y_high:
            return None  # a tripod cannot stand on the table
        t_table_cam = camera_pose_looking_at(position, aim_table)
        points = np.vstack([self._required, self._board_points])
        pixels, depth = project_points(points, t_table_cam, self._intrinsics)
        if (depth < 0.15).any():
            return None
        width, height = self._intrinsics.image_width, self._intrinsics.image_height
        margin = np.minimum.reduce(
            [pixels[:, 0] / width, 1 - pixels[:, 0] / width, pixels[:, 1] / height, 1 - pixels[:, 1] / height]
        ).min()
        if margin < 0.02:
            return None
        board_px, board_depth = project_points(self._board_points, t_table_cam, self._intrinsics)
        board_width_px = float(np.linalg.norm(board_px[1] - board_px[0]))
        ray = position - self._board_centre
        incidence = np.arccos(np.clip(ray[2] / np.linalg.norm(ray), -1, 1))
        centre_depth = float(np.linalg.norm(position - self._work_centre_table))
        hand_width_px = self._intrinsics.fx * 0.09 / centre_depth
        to_camera = position - self._work_centre_table
        if np.arcsin(to_camera[2] / np.linalg.norm(to_camera)) < self._min_elevation_rad:
            return None
        if (
            board_width_px < self._min_board_width_px
            or incidence > self._max_incidence_rad
            or hand_width_px < self._min_hand_width_px
            or board_depth.min() < 0.2
        ):
            return None
        forward = t_table_cam[:3, 2]
        return CameraPlacement(
            position_table=position,
            aim_table=np.asarray(aim_table, float),
            t_table_cam=t_table_cam,
            min_margin=float(margin),
            board_width_px=board_width_px,
            hand_width_px=float(hand_width_px),
            elevation_deg=float(np.degrees(np.arcsin(-forward[2]))),
            height_m=float(position[2]),
            horizontal_fov_deg=float(
                2.0 * np.degrees(np.arctan2(self._intrinsics.image_width / 2.0, self._intrinsics.fx))
            ),
        )

    def search(self, step_m: float = 0.10) -> list[CameraPlacement]:
        """Every feasible pose on a grid, best first.

        Args:
            step_m: grid spacing of lens positions in x, y and z.

        Returns:
            Feasible placements sorted by decreasing margin; empty if none exists.
        """
        aims = [
            self._work_centre_table + np.array([0.0, 0.0, 0.05]),
            0.5 * (self._work_centre_table + self._board_centre) + np.array([0.0, 0.0, 0.05]),
        ]
        table = self._setup.table
        xs = np.arange(table.left_edge_x_m - 1.0, table.right_edge_x_m + 1.0, step_m)
        ys = np.arange(table.near_edge_y_m - 0.9, table.far_edge_y_m + 0.9, step_m)
        zs = np.arange(0.35, 1.15, step_m)
        found = []
        for x in xs:
            for y in ys:
                for z in zs:
                    for aim in aims:
                        placement = self.evaluate(np.array([x, y, z]), aim)
                        if placement is not None:
                            found.append(placement)
        return sorted(found, key=lambda p: -p.min_margin)


def workspace_rect_table(setup: RealSetup, workspace: Workspace) -> tuple[float, float, float, float]:
    """The work area as an axis-aligned rectangle in the table frame: (x_min, x_max, y_min, y_max).

    The table-to-world yaw is a quarter turn, so the world rectangle stays axis-aligned in the table frame.
    """
    mapping = table_to_world_map(setup, workspace)
    corners = np.array(
        [
            mapping.point_to_table(np.array([x, y, 0.9]))[:2]
            for x in workspace.x_range
            for y in workspace.y_range
        ]
    )
    return (
        float(corners[:, 0].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].min()),
        float(corners[:, 1].max()),
    )


@dataclass(frozen=True)
class HandStart:
    """Where the hand is when a clip starts. Measured from the middle of the work area, table axes.

    Attributes:
        name: short label used in the plan.
        description: what to do, in words.
        offset_xyz_m: hand position relative to the work-area centre (x right, y away from the person, z up).
    """

    name: str
    description: str
    offset_xyz_m: tuple[float, float, float]


#: Starting poses spread around the workspace. The robot never sees these (it starts from home); they
#: only make the approach directions and heights in the data differ from clip to clip.
HAND_STARTS = (
    HandStart("low-near-left", "hand flat on the table, near the left edge", (-0.40, -0.18, 0.03)),
    HandStart("low-near-right", "hand flat on the table, near the right edge", (0.40, -0.18, 0.03)),
    HandStart(
        "high-over-workspace", "hand raised, held above the middle of the work area", (0.0, -0.05, 0.28)
    ),
    HandStart("low-far-right", "hand resting on the table at the far right", (0.45, 0.15, 0.03)),
    HandStart("mid-left", "hand held half a forearm above the table at the left", (-0.35, 0.0, 0.15)),
    HandStart("mid-near", "hand held at chest height right in front of you", (0.0, -0.22, 0.20)),
)


@dataclass(frozen=True)
class PlannedClip:
    """One clip of the session.

    Attributes:
        index: 1-based clip number, used in the file name.
        bowl_xy_table: where the bowl centre goes, table frame.
        plate_xy_table: where the plate centre goes, table frame.
        hand_start: where the hand starts.
        grasp_side: which side of the bowl rim to pinch, as the person sees it ("left" or "right").
    """

    index: int
    bowl_xy_table: np.ndarray
    plate_xy_table: np.ndarray
    hand_start: HandStart
    grasp_side: str

    def from_table_edges_m(self, xy_table: np.ndarray, setup: RealSetup) -> tuple[float, float]:
        """(distance from the left table edge, distance from the near table edge) of a table-frame point."""
        return (
            float(xy_table[0] - setup.table.left_edge_x_m),
            float(xy_table[1] - setup.table.near_edge_y_m),
        )


class CapturePlanner:
    """Draws the layouts to record so the work area is covered evenly and every one is runnable in the sim."""

    def __init__(self, setup: RealSetup, workspace: Workspace, seed: int = 0) -> None:
        """Bind to the setup.

        Args:
            setup: measured real setup.
            workspace: world rectangle of object centres; layouts are drawn inside it.
            seed: for the low-discrepancy sequence and the shuffles.
        """
        self._setup = setup
        self._workspace = workspace
        self._mapping = table_to_world_map(setup, workspace)
        self._seed = seed
        self._sampler = LayoutSampler(workspace, setup.bowl.radius_m, setup.plate.radius_m, seed=seed)

    def plan(self, count: int) -> list[PlannedClip]:
        """Layouts, hand starts and grasp sides for `count` clips.

        Layouts come from a scrambled Halton sequence over (bowl x, bowl y, plate x, plate y), keeping the
        ones whose rims do not touch, so the planned positions spread over the work area instead of
        clumping. Hand starts and grasp sides are balanced and shuffled.

        Args:
            count: number of clips.

        Returns:
            The clips, in recording order.
        """
        halton = qmc.Halton(d=4, scramble=True, seed=self._seed)
        low = np.array([self._workspace.x_range[0], self._workspace.y_range[0]] * 2)
        high = np.array([self._workspace.x_range[1], self._workspace.y_range[1]] * 2)
        layouts: list[SimLayout] = []
        while len(layouts) < count:
            raw = qmc.scale(halton.random(1), low, high)[0]
            layout = SimLayout(bowl_xy=raw[:2], plate_xy=raw[2:])
            if np.linalg.norm(layout.bowl_xy - layout.plate_xy) >= self._sampler.min_centre_distance_m:
                layouts.append(layout)
        rng = np.random.default_rng(self._seed)
        sides = ["left" if i % 2 == 0 else "right" for i in range(count)]
        rng.shuffle(sides)
        starts: list[HandStart] = []
        for number, layout in enumerate(layouts):
            bowl_t, plate_t = (
                self._world_to_table_xy(layout.bowl_xy),
                self._world_to_table_xy(layout.plate_xy),
            )
            # Cycle through the starts so they stay balanced, skipping ones that would sit on an object.
            for shift in range(len(HAND_STARTS)):
                start = HAND_STARTS[(number + shift) % len(HAND_STARTS)]
                xy = self.hand_start_table(start)[:2]
                clear = start.offset_xyz_m[2] > 0.1 or (
                    np.linalg.norm(xy - plate_t) > self._setup.plate.radius_m + 0.04
                    and np.linalg.norm(xy - bowl_t) > self._setup.bowl.radius_m + 0.04
                )
                if clear:
                    break
            starts.append(start)
        clips = []
        for number, (layout, start, side) in enumerate(zip(layouts, starts, sides, strict=True), start=1):
            clips.append(
                PlannedClip(
                    index=number,
                    bowl_xy_table=self._world_to_table_xy(layout.bowl_xy),
                    plate_xy_table=self._world_to_table_xy(layout.plate_xy),
                    hand_start=start,
                    grasp_side=side,
                )
            )
        return clips

    def hand_start_table(self, start: HandStart) -> np.ndarray:
        """Table-frame position of a hand start, relative to the centre of the work area."""
        centre = self._mapping.point_to_table(np.array([*self._workspace.centre, 0.9]))
        return np.array([centre[0], centre[1], 0.0]) + np.asarray(start.offset_xyz_m)

    def _world_to_table_xy(self, xy_world: np.ndarray) -> np.ndarray:
        return self._mapping.point_to_table(np.array([xy_world[0], xy_world[1], 0.9]))[:2]
