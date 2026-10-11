"""Live preview window (Tk) that also reports keyboard state, for watching and driving the sim."""

from __future__ import annotations

import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class LiveWindow:
    """A window showing BGR frames, which doubles as a `teleop.KeySource`.

    Tk rather than OpenCV's `imshow`: the environment's `cv2` has no GUI backend (the
    headless wheel shadows it), and Tk delivers key-*release* events, which held-key
    driving needs. Key names are Tk keysyms lower-cased (`up`, `space`, `escape`, `w`, ...).
    """

    #: A release is ignored if the key is pressed again within this time: X11 auto-repeat
    #: emits release/press pairs while a key is held.
    _REPEAT_GRACE_S = 0.06

    def __init__(self, title: str = "hurobo teleop") -> None:
        """Open the window (needs a display)."""
        self._root = tk.Tk()
        self._root.title(title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._photo: ImageTk.PhotoImage | None = None
        self._down: set[str] = set()
        self._released_at: dict[str, float] = {}
        self._fresh: set[str] = set()
        self._closed = False
        self._root.bind("<KeyPress>", self._on_press)
        self._root.bind("<KeyRelease>", self._on_release)
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
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(image)  # keep a reference or Tk drops the image
        self._label.configure(image=self._photo)
        self._root.update()

    def pressed(self) -> set[str]:
        """See `teleop.KeySource`."""
        now = time.monotonic()
        for key, t in list(self._released_at.items()):
            if now - t > self._REPEAT_GRACE_S:
                self._down.discard(key)
                del self._released_at[key]
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

    def _on_press(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        self._released_at.pop(key, None)
        if key not in self._down:
            self._fresh.add(key)
        self._down.add(key)

    def _on_release(self, event: tk.Event) -> None:
        self._released_at[event.keysym.lower()] = time.monotonic()
