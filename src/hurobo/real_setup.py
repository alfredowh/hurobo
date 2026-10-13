"""The hand-measured real setup (table, ChArUco board, bowl, plate) loaded from `measurements.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

#: Where the first real setup was measured; every default in the experiment code points here.
DEFAULT_MEASUREMENTS = Path("data/raw/setup_01/measurements.yaml")


@dataclass(frozen=True)
class RoundObjectSpec:
    """A rotationally symmetric object resting on the table (bowl, plate).

    Attributes:
        name: role in the task, "bowl" or "plate".
        diameter_m: outer diameter at the rim.
        height_m: height from the tabletop to the highest point.
    """

    name: str
    diameter_m: float
    height_m: float

    @property
    def radius_m(self) -> float:
        """Outer radius."""
        return self.diameter_m / 2.0


@dataclass(frozen=True)
class TableSpec:
    """The real table, in the table frame (board corner = origin, x right, y toward the far edge, z up).

    Attributes:
        width_m: length along table-frame x.
        depth_m: length along table-frame y.
        left_edge_x_m: x of the left edge.
        right_edge_x_m: x of the right edge.
        far_edge_y_m: y of the far edge (the one the board sits against).
        near_edge_y_m: y of the near edge (where the person sits); negative.
    """

    width_m: float
    depth_m: float
    left_edge_x_m: float
    right_edge_x_m: float
    far_edge_y_m: float
    near_edge_y_m: float

    @property
    def centre_xy_m(self) -> np.ndarray:
        """shape (2,), the table centre in the table frame."""
        return np.array(
            [(self.left_edge_x_m + self.right_edge_x_m) / 2.0, (self.far_edge_y_m + self.near_edge_y_m) / 2.0]
        )


@dataclass(frozen=True)
class BoardSpec:
    """The ChArUco board that defines the table frame, shown on a tablet.

    Attributes:
        squares_x: squares along the board's long side.
        squares_y: squares along the board's short side.
        square_length_m: measured on the screen.
        marker_length_m: ArUco marker side inside a square.
        thickness_m: height of the screen above the tabletop; table-frame z = 0 is the tabletop.
    """

    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    thickness_m: float

    @property
    def width_m(self) -> float:
        """Grid extent along table-frame x."""
        return self.squares_x * self.square_length_m

    @property
    def height_m(self) -> float:
        """Grid extent along table-frame y."""
        return self.squares_y * self.square_length_m

    def make_board(self, dictionary: int = cv2.aruco.DICT_4X4_50) -> cv2.aruco.CharucoBoard:
        """The OpenCV board definition; detection and calibration must use the same one."""
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            cv2.aruco.getPredefinedDictionary(dictionary),
        )


@dataclass(frozen=True)
class RealSetup:
    """Everything measured by hand about the recording setup.

    Attributes:
        table: table extents in the table frame.
        board: the ChArUco board that defines the frame.
        bowl: the real bowl.
        plate: the real plate.
    """

    table: TableSpec
    board: BoardSpec
    bowl: RoundObjectSpec
    plate: RoundObjectSpec

    @classmethod
    def load(cls, path: Path = DEFAULT_MEASUREMENTS) -> RealSetup:
        """Read a `measurements.yaml`.

        Args:
            path: the file; lengths in metres.

        Returns:
            The parsed setup.

        Raises:
            KeyError: if a required measurement is missing from the file.
        """
        raw = yaml.safe_load(Path(path).read_text())
        table, charuco = raw["table"], raw["charuco"]
        return cls(
            table=TableSpec(
                width_m=float(table["width_m"]),
                depth_m=float(table["depth_m"]),
                left_edge_x_m=float(table["left_edge_x_m"]),
                right_edge_x_m=float(table["right_edge_x_m"]),
                far_edge_y_m=float(table["top_edge_y_m"]),
                near_edge_y_m=float(table["near_edge_y_m"]),
            ),
            board=BoardSpec(
                squares_x=int(charuco["squares_x"]),
                squares_y=int(charuco["squares_y"]),
                square_length_m=float(charuco["square_length_m"]),
                marker_length_m=float(charuco["marker_length_m"]),
                thickness_m=float(charuco["tablet_thickness_m"]),
            ),
            bowl=RoundObjectSpec("bowl", float(raw["bowl"]["diameter_m"]), float(raw["bowl"]["height_m"])),
            plate=RoundObjectSpec(
                "plate", float(raw["plate"]["diameter_m"]), float(raw["plate"]["height_m"])
            ),
        )
