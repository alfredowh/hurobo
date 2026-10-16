"""Local web UI to review recorded clips and set, per clip, which stretch is kept (writes `cuts.json`)."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from hurobo.clip_clean import (
    ANALYSIS_DIR,
    VIDEO_SUFFIXES,
    ClipAnalyzer,
    ClipCleaner,
    ClipCut,
    CutStore,
    FrameFilter,
    find_clips,
)
from hurobo.utils.hand_landmarks import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP

_PAGE = Path(__file__).with_name("clip_cutter.html")
_PROXY_WIDTH_PX = 960


def _nullable(values: np.ndarray, digits: int) -> list[float | None]:
    """JSON-safe list: NaN becomes null."""
    return [None if np.isnan(v) else round(float(v), digits) for v in values]


class ClipCutterApp:
    """Serves the cutter page and its API; analysis results are cached next to the clips."""

    def __init__(self, analyzer: ClipAnalyzer, cleaner: ClipCleaner | None = None) -> None:
        """Hold the analyzer and the cleaner.

        Args:
            analyzer: produces the per-frame signals of a clip (cached on disk).
            cleaner: proposes the automatic cut; defaults to `ClipCleaner()`.
        """
        self._analyzer = analyzer
        self._cleaner = cleaner or ClipCleaner()
        self._frame_filter = FrameFilter(self._cleaner.config)
        self._stores: dict[Path, CutStore] = {}
        self._lock = threading.Lock()

    def serve(self, port: int = 8765) -> ThreadingHTTPServer:
        """Create the HTTP server (localhost only); call `serve_forever` on it."""
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's name
                app._get(self)

            def do_POST(self) -> None:  # noqa: N802
                app._post(self)

            def log_message(self, *_args: object) -> None:
                pass

        return ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def store_for(self, clip: Path) -> CutStore:
        """The cuts file next to a clip, loaded once."""
        folder = clip.parent.resolve()
        if folder not in self._stores:
            self._stores[folder] = CutStore.for_folder(folder)
        return self._stores[folder]

    def list_clips(self, location: str) -> list[dict]:
        """Rows for the clip list: path, name and the stored decision if there is one."""
        rows = []
        for clip in find_clips(location):
            cut = self.store_for(clip).get(clip)
            rows.append(
                {
                    "path": str(clip.resolve()),
                    "name": clip.name,
                    "stored": cut is not None,
                    "accepted": None if cut is None else cut.accepted,
                    "source": None if cut is None else cut.source,
                    "start_s": None if cut is None else cut.start_s,
                    "end_s": None if cut is None else cut.end_s,
                }
            )
        return rows

    def clip_payload(self, clip: Path) -> dict:
        """Everything the page needs for one clip: signals, flags, the automatic and the stored cut."""
        with self._lock:
            signals = self._analyzer.analyze_cached(clip)
            self._ensure_proxy(clip)
        segmenter = self._cleaner.segmenter
        auto = self._cleaner.propose(signals)
        stored = self.store_for(clip).get(clip)
        flags = self._frame_filter.flags(signals)
        landmarks = [
            None if np.isnan(points[0, 0]) else [round(float(v), 1) for v in points.reshape(-1)]
            for points in signals.landmarks_px
        ]
        return {
            "name": clip.name,
            "t0": float(signals.times_s[0]),
            "times_s": [round(float(t), 4) for t in signals.times_s],
            "fps": signals.fps,
            "image_size": list(signals.image_size),
            "hand_count": signals.hand_count.tolist(),
            "aperture_cm": _nullable(signals.aperture_m * 100, 2),
            "reprojection_px": _nullable(signals.reprojection_px, 1),
            "flags": {name: mask.astype(int).tolist() for name, mask in flags.items()},
            "usable": self._frame_filter.usable(signals).astype(int).tolist(),
            "closed": self._cleaner.closed_signal(signals).astype(int).tolist(),
            "landmarks": landmarks,
            "connections": [list(pair) for pair in HAND_CONNECTIONS],
            "tips": [THUMB_TIP, INDEX_TIP],
            "thresholds_cm": [100 * segmenter.close_threshold_m, 100 * segmenter.open_threshold_m],
            "layout_seen": signals.layout_seen,
            "auto": _cut_json(auto),
            "stored": None if stored is None else _cut_json(stored),
        }

    def save_cut(self, clip: Path, start_s: float | None, end_s: float | None, accepted: bool) -> None:
        """Store the user's decision for a clip."""
        cut = ClipCut(
            start_s,
            end_s,
            accepted,
            source="manual",
            reasons=[] if accepted else ["rejected by hand"],
            notes=["edited in the cutter"],
        )
        self.store_for(clip).set(clip, cut)

    def reset_cut(self, clip: Path) -> None:
        """Forget the stored decision (the page then shows the automatic proposal again)."""
        self.store_for(clip).remove(clip)

    @staticmethod
    def proxy_path(clip: Path) -> Path:
        """Browser-playable copy of a clip (H.264, small, a key frame every few frames for quick seeking)."""
        return clip.parent / ANALYSIS_DIR / f"{clip.stem}.proxy.mp4"

    def _ensure_proxy(self, clip: Path) -> None:
        proxy = self.proxy_path(clip)
        if proxy.exists() and proxy.stat().st_mtime >= clip.stat().st_mtime:
            return
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; needed to make a browser-playable copy")
        proxy.parent.mkdir(exist_ok=True)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(clip), "-an", "-vf", f"scale={_PROXY_WIDTH_PX}:-2"]
            + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "5", "-pix_fmt", "yuv420p"]
            + ["-vsync", "0", "-movflags", "+faststart", str(proxy)],
            check=True,
        )

    def _clip_from_query(self, handler: BaseHTTPRequestHandler, query: dict[str, list[str]]) -> Path | None:
        """The clip named in `?path=`, only if it is an existing video file."""
        clip = Path(query.get("path", [""])[0]).expanduser()
        if clip.suffix.lower() not in VIDEO_SUFFIXES or not clip.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "not a video file")
            return None
        return clip

    def _get(self, handler: BaseHTTPRequestHandler) -> None:
        url = urlparse(handler.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                _send(handler, _PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/clips":
                _send_json(handler, {"clips": self.list_clips(query.get("location", [""])[0])})
            elif url.path == "/api/clip":
                clip = self._clip_from_query(handler, query)
                if clip is not None:
                    _send_json(handler, self.clip_payload(clip))
            elif url.path == "/video":
                clip = self._clip_from_query(handler, query)
                if clip is not None:
                    self._send_file(handler, self.proxy_path(clip))
            else:
                handler.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:  # report to the page instead of dropping the connection
            _send_json(
                handler, {"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _post(self, handler: BaseHTTPRequestHandler) -> None:
        url = urlparse(handler.path)
        body = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))) or b"{}")
        clip = Path(body.get("path", "")).expanduser()
        if clip.suffix.lower() not in VIDEO_SUFFIXES or not clip.is_file():
            handler.send_error(HTTPStatus.NOT_FOUND, "not a video file")
            return
        if url.path == "/api/cut":
            self.save_cut(clip, body.get("start_s"), body.get("end_s"), bool(body.get("accepted")))
        elif url.path == "/api/reset":
            self.reset_cut(clip)
        else:
            handler.send_error(HTTPStatus.NOT_FOUND)
            return
        _send_json(handler, {"ok": True})

    @staticmethod
    def _send_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
        """Serve a file with HTTP range support (browsers need it to seek in a video)."""
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = handler.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            first, _, last = range_header[6:].partition("-")
            start = int(first) if first else 0
            end = int(last) if last else size - 1
        end = min(end, size - 1)
        handler.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
        handler.send_header("Content-Type", "video/mp4")
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(end - start + 1))
        if range_header:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    handler.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


def _cut_json(cut: ClipCut) -> dict:
    return {
        "start_s": cut.start_s,
        "end_s": cut.end_s,
        "accepted": cut.accepted,
        "source": cut.source,
        "reasons": cut.reasons,
        "notes": cut.notes,
    }


def _send(handler: BaseHTTPRequestHandler, body: bytes, content_type: str, status: int = 200) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _send_json(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    _send(handler, json.dumps(payload).encode(), "application/json", status)
