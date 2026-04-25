"""
Touch injection for scrcpy via SetCursorPos + mouse_event.

Moves the real mouse cursor to the target screen position, then fires
mouse_event() which goes through the Windows system input pipeline —
more reliable for SDL windows than PostMessage.

Frame coordinates match the captured frame (top-left of scrcpy window).
"""
import ctypes
import time

import win32api
import win32con
import win32gui

# Ensure DPI-aware coordinate mapping
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class ScrcpyTouch:

    def __init__(self, hwnd: int, title_bar_h: int = 30):
        self._hwnd = hwnd
        self._tbar = title_bar_h

    # ── screen coordinate helper ──────────────────────────────────────────────

    def _screen(self, fx: int, fy: int) -> tuple[int, int]:
        """Frame coords → screen coords using current window position."""
        rect = win32gui.GetWindowRect(self._hwnd)
        return rect[0] + fx, rect[1] + fy

    def _focus(self) -> None:
        """Bring scrcpy window to foreground before interacting."""
        try:
            if win32gui.GetForegroundWindow() != self._hwnd:
                win32gui.SetForegroundWindow(self._hwnd)
                time.sleep(0.08)
        except Exception:
            pass

    # ── low-level events ──────────────────────────────────────────────────────

    def down(self, fx: int, fy: int) -> None:
        sx, sy = self._screen(fx, fy)
        self._focus()
        win32api.SetCursorPos((sx, sy))
        time.sleep(0.02)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def move(self, fx: int, fy: int) -> None:
        sx, sy = self._screen(fx, fy)
        win32api.SetCursorPos((sx, sy))

    def up(self, fx: int, fy: int) -> None:
        sx, sy = self._screen(fx, fy)
        win32api.SetCursorPos((sx, sy))
        time.sleep(0.01)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    # ── gestures ──────────────────────────────────────────────────────────────

    def tap(self, fx: int, fy: int, hold_ms: int = 80) -> None:
        self.down(fx, fy)
        time.sleep(hold_ms / 1000)
        self.up(fx, fy)

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 400, steps: int = 30) -> None:
        self.down(x1, y1)
        delay = duration_ms / 1000 / max(steps, 1)
        for i in range(1, steps + 1):
            t = i / steps
            self.move(int(x1 + (x2 - x1) * t),
                      int(y1 + (y2 - y1) * t))
            time.sleep(delay)
        self.up(x2, y2)

    def drag_start(self, fx: int, fy: int) -> None:
        self.down(fx, fy)

    def drag_to(self, fx: int, fy: int, delay_ms: int = 16) -> None:
        self.move(fx, fy)
        time.sleep(delay_ms / 1000)

    def drag_end(self, fx: int, fy: int) -> None:
        self.up(fx, fy)
