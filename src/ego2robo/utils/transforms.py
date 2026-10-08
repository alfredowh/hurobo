"""SE(3) pose helpers and frame conventions shared across the real2sim pipeline.

Every pose in this codebase is a 4x4 homogeneous transform, named `T_<parent>_<child>`
(e.g. `T_table_bowl` maps a point in the bowl frame to the table frame). Never pass a
bare rotation or translation between modules without a transform-shaped container: frame
bugs are the main time-sink in retargeting (see CLAUDE.md), and the naming convention is
the main defense against them.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(rotation: Rotation, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a rotation and a translation.

    Args:
        rotation: scipy Rotation representing R_parent_child.
        translation: shape (3,), the child origin expressed in the parent frame.

    Returns:
        4x4 float64 array T_parent_child.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation
    return transform


def invert_transform(t_parent_child: np.ndarray) -> np.ndarray:
    """Invert a homogeneous transform: T_parent_child -> T_child_parent.

    Args:
        t_parent_child: 4x4 transform.

    Returns:
        4x4 transform, the algebraic inverse (exploits the rotation's orthogonality
        rather than a general matrix inverse, for numerical stability).
    """
    r = t_parent_child[:3, :3]
    p = t_parent_child[:3, 3]
    t_child_parent = np.eye(4, dtype=np.float64)
    t_child_parent[:3, :3] = r.T
    t_child_parent[:3, 3] = -r.T @ p
    return t_child_parent


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Chain transforms left to right: compose_transforms(T_a_b, T_b_c) -> T_a_c.

    Args:
        *transforms: two or more 4x4 transforms, each the child frame of the last.

    Returns:
        4x4 transform, the matrix product in the given order.
    """
    result = transforms[0]
    for t in transforms[1:]:
        result = result @ t
    return result


def apply_transform(t_parent_child: np.ndarray, points_child: np.ndarray) -> np.ndarray:
    """Map points from the child frame into the parent frame.

    Args:
        t_parent_child: 4x4 transform.
        points_child: shape (N, 3) or (3,), points expressed in the child frame.

    Returns:
        Points in the parent frame, same shape as the input.
    """
    single = points_child.ndim == 1
    pts = points_child.reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = (t_parent_child @ pts_h.T).T[:, :3]
    return out[0] if single else out


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert an OpenCV solvePnP/Rodrigues (rvec, tvec) pair into T_cam_object.

    Args:
        rvec: shape (3,) or (3, 1), Rodrigues rotation vector from solvePnP.
        tvec: shape (3,) or (3, 1), translation from solvePnP.

    Returns:
        4x4 transform T_cam_object, mapping points in the object frame to the
        camera frame (OpenCV camera convention: +z forward, +y down).
    """
    rotation = Rotation.from_rotvec(np.asarray(rvec).reshape(3))
    return make_transform(rotation, np.asarray(tvec).reshape(3))


def cv_camera_pose_to_mujoco(t_world_cam_cv: np.ndarray) -> np.ndarray:
    """Convert a camera pose from OpenCV convention to MuJoCo convention.

    OpenCV: the camera looks down +z, with +y pointing down the image.
    MuJoCo: the camera looks down -z, with +y pointing up the image.
    So the camera-frame axes differ by R_mj_cv = diag(1, -1, -1), applied on the
    right of the world-frame rotation (it flips the camera's own y and z axes,
    not the world frame it sits in). This is the one conversion the SceneTwin
    (`twin.py`) needs to place a MuJoCo camera at a pose measured with OpenCV.

    Args:
        t_world_cam_cv: 4x4 transform T_world_cam in OpenCV camera convention.

    Returns:
        4x4 transform T_world_cam in MuJoCo camera convention. Same origin;
        only the camera's local axis directions change.
    """
    flip = np.diag([1.0, -1.0, -1.0])
    t_world_cam_mj = t_world_cam_cv.copy()
    t_world_cam_mj[:3, :3] = t_world_cam_cv[:3, :3] @ flip
    return t_world_cam_mj


def transform_to_pos_quat_wxyz(t_parent_child: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a transform into a MuJoCo-style (pos, quat) pair.

    Args:
        t_parent_child: 4x4 transform.

    Returns:
        (pos, quat) where pos has shape (3,) and quat is (w, x, y, z), the
        order `mjModel`/`mjData` fields such as `cam_quat` expect (scipy's
        Rotation uses (x, y, z, w), so the order is swapped here).
    """
    pos = t_parent_child[:3, 3].copy()
    x, y, z, w = Rotation.from_matrix(t_parent_child[:3, :3]).as_quat()
    return pos, np.array([w, x, y, z])
