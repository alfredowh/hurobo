"""Scan the setup_01 bowl and plate clips alone and in combinations (feeds `examples/08_*.ipynb`).

    uv run python scripts/15_scan_clips.py                       # everything, ~1.5 h on this machine
    uv run python scripts/15_scan_clips.py --only bowl_high plate

Single clips run `default_clip_sweep()`; a combination reconstructs the keyframes of several clips of the
same object in one COLMAP model (`combined_clip_sweep`). Each set writes
`data/scans/clips/<set>/<config>/{result.json,splat.ply,heldout_preview.png}`; SfM is cached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hurobo.scan_sweep import ScanSweep, combined_clip_sweep, default_clip_sweep, sweep_table

#: set name -> clip stems under the raw dir. Single clips first: they are the cheaper runs.
CLIP_SETS: dict[str, list[str]] = {
    "bowl_high": ["bowl_high"],
    "bowl_med": ["bowl_med"],
    "bowl_low": ["bowl_low"],
    "plate": ["plate"],
    "bowl_high+low": ["bowl_high", "bowl_low"],
    "bowl_high+med": ["bowl_high", "bowl_med"],
    "bowl_med+low": ["bowl_med", "bowl_low"],
    "bowl_high+med+low": ["bowl_high", "bowl_med", "bowl_low"],
    "plate_low": ["plate_low"],
    "plate_med": ["plate_med"],
    "plate_low+high": ["plate_low", "plate_high"],
    "plate_low+med": ["plate_low", "plate_med"],
}


def main() -> None:
    """Parse args, run the chosen clip sets, print each comparison table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/setup_01"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/scans/clips"))
    parser.add_argument("--holdout-every", type=int, default=10)
    parser.add_argument(
        "--only", nargs="+", default=None, choices=list(CLIP_SETS), help="run just these sets"
    )
    args = parser.parse_args()

    for name, stems in CLIP_SETS.items():
        if args.only and name not in args.only:
            continue
        configs = default_clip_sweep() if len(stems) == 1 else combined_clip_sweep(len(stems))
        videos = [args.raw_dir / f"{stem}.mp4" for stem in stems]
        sweep = ScanSweep(videos, args.out_dir / name.replace("+", "_"), args.holdout_every)
        print(f"\n=== {name} ===")
        print(sweep_table(sweep.run(configs)).to_string(index=False))


if __name__ == "__main__":
    main()
