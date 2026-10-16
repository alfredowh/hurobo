#!/usr/bin/env python
"""Proposes a cut for every recorded clip, before the demos are extracted.

For each clip: drop frames without a hand (start and end), frames with two hands, badly fitted or
edge-clipped hands; keep only the stretch where the pinch is open, closes once and opens again; reject
clips that are too short or whose markers were not seen at the start. The result is `<folder>/cuts.json`,
which `02_extract_demos.py --cuts` reads. Review and adjust it in the UI (`21_clip_cutter.py`).

    uv run python scripts/20_clean_clips.py data/raw/e1 --intrinsics data/camera_intrinsics_s10e.npz

Cuts you edited by hand (source "manual") are kept unless you pass --overwrite-manual. Per-frame signals
are cached next to the clips in `.analysis/`, so a rerun is quick.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hurobo.camera import CameraIntrinsics
from hurobo.clip_clean import ClipAnalyzer, ClipCleaner, CutStore, find_clips
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.recording import RecordingSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("clips", help="a folder, a glob (quote it) or a single video")
    parser.add_argument("--intrinsics", type=Path, default=Path("data/camera_intrinsics_s10e.npz"))
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--cuts", type=Path, default=None, help="default: cuts.json in the clips' folder")
    parser.add_argument("--overwrite-manual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = find_clips(args.clips)
    if not clips:
        raise SystemExit(f"no clips found at {args.clips}")
    intrinsics = CameraIntrinsics.load(args.intrinsics)
    spec = RecordingSpec()
    analyzer = ClipAnalyzer(intrinsics, spec.estimator(intrinsics, RealSetup.load(args.setup)))
    cleaner = ClipCleaner()
    store = CutStore(args.cuts or clips[0].parent / "cuts.json")
    kept = 0
    for clip in clips:
        existing = store.get(clip)
        if existing is not None and existing.source == "manual" and not args.overwrite_manual:
            print(f"[KEEP] {clip.name}: manual cut kept")
            kept += existing.accepted
            continue
        cut = cleaner.propose(analyzer.analyze_cached(clip))
        store.set(clip, cut)
        kept += cut.accepted
        window = f"{cut.start_s:.2f}-{cut.end_s:.2f} s" if cut.window_s else "no window"
        print(
            f"[{'OK  ' if cut.accepted else 'DROP'}] {clip.name}: {window}; "
            + "; ".join(cut.reasons + cut.notes)
        )
    print(f"{kept}/{len(clips)} clips accepted; cuts -> {store.path}")


if __name__ == "__main__":
    main()
