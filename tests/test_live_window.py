"""LiveWindow key handling: a key pressed again after being released is a fresh press."""

from __future__ import annotations

import time
import tkinter as tk

import numpy as np
import pytest

from hurobo.live import LiveWindow


def test_repeated_presses_are_fresh_without_polling_pressed() -> None:
    try:
        window = LiveWindow("test")
    except tk.TclError:
        pytest.skip("no display")
    frame = np.zeros((8, 8, 3), np.uint8)
    try:
        window.show(frame)  # maps the window so it can take key events
        for _ in range(3):
            window.inject_key("c", True)
            window.show(frame)
            window.inject_key("c", False)
            window.show(frame)
            assert window.consume("c")
            time.sleep(0.1)  # longer than the auto-repeat grace period
    finally:
        window.close()
