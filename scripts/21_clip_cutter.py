#!/usr/bin/env python
"""Opens the clip cutter in the browser: review every clip, drag the cut handles, save to `cuts.json`.

    uv run python scripts/21_clip_cutter.py data/raw/e1 --intrinsics data/camera_intrinsics_s10e.npz

Run `20_clean_clips.py` first to get automatic cuts (the page also proposes one for a clip that has none).
The first time a clip is opened it is analysed (about as long as the clip) and a small browser-playable copy
is made in `<folder>/.analysis/`. The server listens on localhost only. Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path
from urllib.parse import quote

from hurobo.camera import CameraIntrinsics
from hurobo.clip_clean import ClipAnalyzer
from hurobo.clip_cutter_app import ClipCutterApp
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.recording import RecordingSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("location", nargs="?", default="", help="folder, glob or video to preload")
    parser.add_argument("--intrinsics", type=Path, default=Path("data/camera_intrinsics_s10e.npz"))
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intrinsics = CameraIntrinsics.load(args.intrinsics)
    estimator = RecordingSpec().estimator(intrinsics, RealSetup.load(args.setup))
    server = ClipCutterApp(ClipAnalyzer(intrinsics, estimator)).serve(args.port)
    url = f"http://127.0.0.1:{args.port}/" + (f"?location={quote(args.location)}" if args.location else "")
    print(f"clip cutter at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
