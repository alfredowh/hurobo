"""The best scanned bowl and plate from the scan study, compared with the measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hurobo.real_setup import RealSetup
from hurobo.splat import splat_dimensions

CLIPS_OBJECTS = Path("data/scans/clips/_objects")


@dataclass(frozen=True)
class BestScans:
    """Paths of the chosen object splats (z-up, base at z = 0); picked in notebook 06.

    Attributes:
        bowl_ply: bowl from the high + low clips scanned together (consistent cameras, closed wall).
        plate_ply: plate from the medium-height clip alone (complete rim, legible pattern).
        plate_filled_ply: the same plate with its gaps closed by the measured shape prior (D-009).
        bowl_sfm_dir: sweep folder of the bowl's SfM, for the camera plot.
        plate_sfm_dir: sweep folder of the plate's SfM.
    """

    bowl_ply: Path = CLIPS_OBJECTS / "bowl_high_low__combined.ply"
    plate_ply: Path = CLIPS_OBJECTS / "plate_med__robust_exhaustive.ply"
    plate_filled_ply: Path = CLIPS_OBJECTS / "plate_med__robust_exhaustive_filled.ply"
    bowl_sfm_dir: Path = Path("data/scans/clips/bowl_high_low")
    plate_sfm_dir: Path = Path("data/scans/clips/plate_med")


def scan_vs_measurement(scans: BestScans, setup: RealSetup) -> pd.DataFrame:
    """Aspect ratio (height / diameter) of each scan against the hand measurement.

    A scan without a metric board has arbitrary units, so only dimensionless ratios can be compared; the
    factor column is what a uniform rescale to the measured diameter would multiply the splat by.

    Args:
        scans: the splat paths.
        setup: measured bowl and plate.

    Returns:
        One row per object.
    """
    rows = []
    for name, ply, real in (
        ("bowl", scans.bowl_ply, setup.bowl),
        ("plate", scans.plate_ply, setup.plate),
        ("plate (shape-prior fill)", scans.plate_filled_ply, setup.plate),
    ):
        diameter, height = splat_dimensions(ply)
        rows.append(
            {
                "object": name,
                "scan diameter (units)": diameter,
                "scan height (units)": height,
                "scan height/diameter": height / diameter,
                "real height/diameter": real.height_m / real.diameter_m,
                "real diameter (cm)": 100 * real.diameter_m,
                "metres per scan unit": real.diameter_m / diameter,
            }
        )
    return pd.DataFrame(rows).round(3)
