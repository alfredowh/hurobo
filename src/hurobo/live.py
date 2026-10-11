"""Live preview window (Tk) that also reports keyboard state, for watching and driving the sim."""

from __future__ import annotations

import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from hurobo.utils.video import fit_in_box


class LiveWindow:
    """A window showing BGR frames, which doubles as a `teleop.KeySource`.

    Tk rather than OpenCV's `imshow`: the environment's `cv2` has no GUI backend (the
    headless wheel shadows it), and Tk delivers key-*release* events, which held-key
    driving needs. Key names are Tk keysyms lower-cased (`up`, `space`, `escape`, `w`, ...).

    The frame is scaled to fit the window (aspect ratio kept, black bars), so the window can be
    resized or made full screen; F11 toggles full screen.
    """

    #: A release is ignored if the key is pressed again within this time: X11 auto-repeat
    #: emits release/press pairs while a key is held.
    _REPEAT_GRACE_S = 0.06

    def __init__(self, title: str = "hurobo teleop", fullscreen: bool = False) -> None:
        """Open the window (needs a display).

        Args:
            title: window title.
            fullscreen: start full screen (F11 toggles it).
        """
        self._root = tk.Tk()
        self._root.title(title)
        self._root.configure(bg="black")
        self._label = tk.Label(self._root, bg="black", borderwidth=0)
        self._label.pack(fill="both", expand=True)
        self._sized = False
        self._fullscreen = fullscreen
        self._root.attributes("-fullscreen", fullscreen)
        self._photo: ImageTk.PhotoImage | None = None
        self._down: set[str] = set()
        self._released_at: dict[str, float] = {}
        self._fresh: set[str] = set()
        self._closed = False
        self._root.bind("<KeyPress>", self._on_press)
        self._root.bind("<KeyRelease>", self._on_release)
        self._root.bind("<F11>", self._toggle_fullscreen)
        self._label.bind("<Button-1>", lambda _event: self._root.focus_force())  # click to regain keys
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.focus_force()

    @property
    def closed(self) -> bool:
        """True once the user closed the window (or `close` was called)."""
        return self._closed

    def show(self, frame_bgr: np.ndarray) -> None:
        """Display a frame and process pending window events.

        Args:
            frame_bgr: shape (H, W, 3) uint8 BGR.
        """
        if self._closed:
            return
        if not self._sized:
            # Fix the initial size once, otherwise Tk resizes the window to every new image.
            width = min(frame_bgr.shape[1], self._root.winfo_screenwidth())
            height = min(frame_bgr.shape[0], self._root.winfo_screenheight())
            self._root.geometry(f"{width}x{height}")
            self._root.update()
            self._root.focus_force()  # the resize can leave the window without keyboard focus
            self._sized = True
        box_width, box_height = self._label.winfo_width(), self._label.winfo_height()
        if box_width > 1 and box_height > 1:  # 1x1 until the window is mapped
            frame_bgr = fit_in_box(frame_bgr, box_width, box_height)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(image)  # keep a reference or Tk drops the image
        self._label.configure(image=self._photo)
        self._root.update()

    def process_events(self) -> None:
        """Handle pending window and key events without changing the picture.

        Call it while waiting for the first camera frame, so the window stays responsive.
        """
        if not self._closed:
            self._root.update()

    def pressed(self) -> set[str]:
        """See `teleop.KeySource`."""
        self._expire_releases()
        return set(self._down)

    def consume(self, key: str) -> bool:
        """See `teleop.KeySource`."""
        if key in self._fresh:
            self._fresh.discard(key)
            return True
        return False

    def inject_key(self, keysym: str, pressed: bool) -> None:
        """Deliver a synthetic key event as if typed, for tests and scripted demos.

        Args:
            keysym: Tk keysym, e.g. `Up`, `space`, `Escape`, `w`.
            pressed: True for key-down, False for key-up.
        """
        self._root.event_generate("<KeyPress>" if pressed else "<KeyRelease>", keysym=keysym)

    def close(self) -> None:
        """Destroy the window. Safe to call twice."""
        if not self._closed:
            self._closed = True
            self._root.destroy()

    def _toggle_fullscreen(self, _event: tk.Event) -> None:
        self._fullscreen = not self._fullscreen
        self._root.attributes("-fullscreen", self._fullscreen)

    def _expire_releases(self) -> None:
        """Forget keys released longer ago than the auto-repeat grace period."""
        now = time.monotonic()
        for key, t in list(self._released_at.items()):
            if now - t > self._REPEAT_GRACE_S:
                self._down.discard(key)
                del self._released_at[key]

    def _on_press(self, event: tk.Event) -> None:
        # Expire here too: a loop that only calls `consume` (calibration) never calls `pressed`,
        # and a stale "down" entry would make every later press of that key look like a repeat.
        self._expire_releases()
        key = event.keysym.lower()
        self._released_at.pop(key, None)
        if key not in self._down:
            self._fresh.add(key)
        self._down.add(key)

    def _on_release(self, event: tk.Event) -> None:
        self._released_at[event.keysym.lower()] = time.monotonic()
