"""MediaPipe's 21-point hand model: landmark names, bone connectivity and the pinch landmarks."""

from __future__ import annotations

import numpy as np

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_JOINTS = {"thumb": ("cmc", "mcp", "ip", "tip"), "other": ("mcp", "pip", "dip", "tip")}

LANDMARK_NAMES: tuple[str, ...] = ("wrist",) + tuple(
    f"{finger}_{joint}" for finger in _FINGERS for joint in _JOINTS["thumb" if finger == "thumb" else "other"]
)

WRIST = 0
THUMB_TIP = LANDMARK_NAMES.index("thumb_tip")
INDEX_TIP = LANDMARK_NAMES.index("index_tip")

# Same topology as MediaPipe's HAND_CONNECTIONS: wrist-to-finger-base and base-to-base links
# make the palm, then each finger is a chain base -> ... -> tip.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),  # palm
    (1, 2), (2, 3), (3, 4),  # thumb
    (5, 6), (6, 7), (7, 8),  # index
    (9, 10), (10, 11), (11, 12),  # middle
    (13, 14), (14, 15), (15, 16),  # ring
    (17, 18), (18, 19), (19, 20),  # pinky
)  # fmt: skip

#: Landmark index ranges of the thumb and index finger, which the pinch is made of.
THUMB_CHAIN = tuple(range(1, 5))
INDEX_CHAIN = tuple(range(5, 9))


def polyline_with_breaks(points: np.ndarray) -> np.ndarray:
    """Hand skeleton as one NaN-separated polyline, so a single line artist draws every bone.

    Args:
        points: shape (21, D) landmark coordinates in any frame.

    Returns:
        shape (3 * len(HAND_CONNECTIONS), D); a NaN row ends each bone.
    """
    segments = []
    for start, end in HAND_CONNECTIONS:
        segments.extend([points[start], points[end], np.full(points.shape[1], np.nan)])
    return np.array(segments)
