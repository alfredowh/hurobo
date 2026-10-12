"""Run the scan pipeline under several parameter sets on one clip (feeds `examples/08_*.ipynb`).

    uv run python scripts/14_scan_param_sweep.py data/raw/setup_01/bowl_1angle.mp4 --name bowl_1angle

Each config writes `data/scans/sweeps/<name>/<config>/{result.json,splat.ply,heldout_preview.png}`.
SfM is cached per (frame count, options); re-running only redoes what changed. The default set took about
30 min on a 25 s clip here (exhaustive matching alone is ~6 min: it is O(N^2) on CPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hurobo.scan_sweep import ScanSweep, default_bowl_sweep, sweep_table


def main() -> None:
    """Parse args, run the chosen configs, print the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--name", required=True, help="sweep folder name under --out-dir")
    parser.add_argument("--out-dir", type=Path, default=Path("data/scans/sweeps"))
    parser.add_argument(
        "--holdout-every", type=int, default=10, help="withhold every N-th keyframe to score on"
    )
    parser.add_argument("--only", nargs="+", default=None, help="run just these config names")
    args = parser.parse_args()

    configs = default_bowl_sweep()
    if args.only:
        unknown = set(args.only) - {c.name for c in configs}
        if unknown:
            parser.error(f"unknown config(s) {sorted(unknown)}; choose from {[c.name for c in configs]}")
        configs = [c for c in configs if c.name in args.only]

    results = ScanSweep(args.video, args.out_dir / args.name, args.holdout_every).run(configs)
    print(sweep_table(results).to_string(index=False))


if __name__ == "__main__":
    main()
