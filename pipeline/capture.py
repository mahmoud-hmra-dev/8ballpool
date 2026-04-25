"""
Screen capture via Win32 PrintWindow API.

PrintWindow grabs the window's own GDI content directly — it works even
when Chrome is covered by VS Code, a terminal, or any other window.

PW_RENDERFULLCONTENT = 0x2  →  forces GPU-accelerated Chrome to blit its
composited output into our DC. Falls back to mss if PrintWindow fails.

GDI objects (DC + bitmap) are created once and reused every frame,
saving ~10 Win32 calls per frame compared to creating them fresh each time.
"""
import logging
import threading

import cv2
import mss as _mss_module
import numpy as np
import win32gui
import win32ui
from ctypes import windll

from config import BROWSER_TOP_OFFSET

log = logging.getLogger(__name__)


class ScreenCapture:

    def __init__(self):
        self._hwnd:     int | None = None
        self._win_rect: tuple | None = None
        self._region:   tuple | None = None

        # Cached GDI objects — created once in _init_dc, reused forever
        self._save_dc = None
        self._bmp     = None
        self._win_w   = 0
        self._win_h   = 0

        # Per-thread mss instance (mss is not thread-safe)
        self._local = threading.local()

    # ── window discovery ──────────────────────────────────────────────────────

    def find_game_window(self) -> tuple | None:
        """
        Enumerate visible windows looking for Chrome with "8 Ball Pool" in the
        title (priority 10) or any Chrome window (priority 1).
        Restores minimized windows automatically.
        Returns (x1, y1, x2, y2) of the capture region, or None.
        """
        found     = []   # (priority, hwnd, rect)
        minimized = []   # (priority, hwnd, title)

        def _cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if "8 Ball Pool" in title:
                prio = 10
            elif "Chrome" in title or "chrome" in title:
                prio = 1
            else:
                return
            if win32gui.IsIconic(hwnd):
                minimized.append((prio, hwnd, title))
            else:
                found.append((prio, hwnd, win32gui.GetWindowRect(hwnd)))

        win32gui.EnumWindows(_cb, None)

        if not found:
            return self._try_restore(minimized)

        found.sort(reverse=True)
        _, hwnd, rect = found[0]
        log.info("Window: '%s'  rect=%s", win32gui.GetWindowText(hwnd), rect)
        return self._init_region(hwnd, rect, top_offset=BROWSER_TOP_OFFSET)

    def find_scrcpy_window(self) -> tuple | None:
        """
        Find the scrcpy phone-mirror display window (SDL_app class).
        scrcpy uses SDL2 — its display window class is always 'SDL_app'.
        The title is usually the device model name (e.g. 'SM_S938B').
        No browser offset — the game fills the entire scrcpy window.
        Returns (x1, y1, x2, y2) of the capture region, or None.
        """
        import win32api
        import win32con

        found     = []
        minimized = []

        def _cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            # scrcpy display window class is SDL_app; skip console/terminal windows
            cls = win32gui.GetClassName(hwnd)
            if cls != "SDL_app":
                return
            title = win32gui.GetWindowText(hwnd)
            if win32gui.IsIconic(hwnd):
                minimized.append((1, hwnd, title))
            else:
                found.append((1, hwnd, win32gui.GetWindowRect(hwnd)))

        win32gui.EnumWindows(_cb, None)

        if not found:
            if not minimized:
                log.warning("No SDL_app window found — is scrcpy running?")
                return None
            minimized.sort(reverse=True)
            _, hwnd, title = minimized[0]
            log.info("Found minimized scrcpy SDL window: '%s' — restoring...", title)
            try:
                win32gui.ShowWindow(hwnd, 9)
                win32gui.SetForegroundWindow(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                if not win32gui.IsIconic(hwnd) and rect[2] > rect[0] and rect[3] > rect[1]:
                    return self._init_region(hwnd, rect, top_offset=0)
            except Exception as e:
                log.warning("Could not restore scrcpy window: %s", e)
            return None

        found.sort(reverse=True)
        _, hwnd, rect = found[0]
        log.info("scrcpy SDL window: '%s'  rect=%s", win32gui.GetWindowText(hwnd), rect)
        return self._init_region(hwnd, rect, top_offset=0)

    def _try_restore(self, minimized: list) -> tuple | None:
        if not minimized:
            return None
        minimized.sort(reverse=True)
        _, hwnd, title = minimized[0]
        log.info("Found minimized Chrome: '%s' — restoring...", title)
        try:
            win32gui.ShowWindow(hwnd, 9)   # SW_RESTORE = 9
            win32gui.SetForegroundWindow(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            if not win32gui.IsIconic(hwnd) and rect[2] > rect[0] and rect[3] > rect[1]:
                return self._init_region(hwnd, rect, top_offset=BROWSER_TOP_OFFSET)
        except Exception as e:
            log.warning("Could not restore window: %s", e)
        return None

    def _init_region(self, hwnd: int, rect: tuple, top_offset: int = BROWSER_TOP_OFFSET) -> tuple:
        self._hwnd     = hwnd
        self._win_rect = rect
        x1, y1, x2, y2 = rect
        self._region = (x1, y1 + top_offset, x2, y2)
        return self._region

    # ── GDI cache ─────────────────────────────────────────────────────────────

    def _init_dc(self):
        """Allocate GDI DC and bitmap sized to the full window. Called once."""
        wx1, wy1, wx2, wy2 = self._win_rect
        win_w, win_h = wx2 - wx1, wy2 - wy1

        hwnd_dc = win32gui.GetWindowDC(self._hwnd)
        mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, win_w, win_h)
        save_dc.SelectObject(bmp)
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self._hwnd, hwnd_dc)

        self._save_dc = save_dc
        self._bmp     = bmp
        self._win_w   = win_w
        self._win_h   = win_h

    # ── capture ───────────────────────────────────────────────────────────────

    def capture(self, region: tuple = None) -> np.ndarray | None:
        """
        Capture the Chrome window, even when it is behind other windows.
        Returns a BGR NumPy array, or None on failure.
        """
        r = region or self._region
        if r is None or self._hwnd is None:
            return None

        if self._save_dc is None:
            self._init_dc()

        x1, y1, x2, y2 = r
        wx1, wy1 = self._win_rect[0], self._win_rect[1]

        try:
            ok = windll.user32.PrintWindow(self._hwnd, self._save_dc.GetSafeHdc(), 2)
            if not ok:
                log.warning("PrintWindow returned 0 — falling back to mss")
                return self._mss_capture(r)

            raw  = self._bmp.GetBitmapBits(True)
            full = np.frombuffer(raw, dtype="uint8").reshape(self._win_h, self._win_w, 4)

            cx1 = max(0, x1 - wx1);  cy1 = max(0, y1 - wy1)
            cx2 = min(self._win_w, x2 - wx1)
            cy2 = min(self._win_h, y2 - wy1)

            return cv2.cvtColor(full[cy1:cy2, cx1:cx2].copy(), cv2.COLOR_BGRA2BGR)

        except Exception as e:
            log.error("PrintWindow error: %s", e)
            self._save_dc = None   # force re-init next frame
            return self._mss_capture(r)

    def _mss_capture(self, region: tuple) -> np.ndarray | None:
        """Thread-local mss fallback (used when PrintWindow fails)."""
        if getattr(self._local, "sct", None) is None:
            self._local.sct = _mss_module.mss()
        x1, y1, x2, y2 = region
        try:
            raw = self._local.sct.grab(
                {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1})
            return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        except Exception as e:
            log.error("mss error: %s", e)
            self._local.sct = None
            return None

    # ── async capture ─────────────────────────────────────────────────────────

    def start_async(self, region: tuple) -> None:
        """
        Spawn a background thread that calls capture() in a tight loop.
        The pipeline reads latest_frame() instead of calling capture() directly,
        so it is never stalled waiting for PrintWindow to finish.

        PrintWindow on a 2560×1440 window takes ~35 ms — making it async
        breaks this bottleneck and lets the pipeline run at 60+ FPS.
        """
        self._async_frame: np.ndarray | None = None
        self._async_lock  = threading.Lock()
        self._region      = region
        t = threading.Thread(
            target=self._capture_loop, daemon=True, name="capture-worker")
        t.start()
        log.info("async capture thread started")

    def latest_frame(self) -> np.ndarray | None:
        """Return the most recently captured frame (may be None on first call)."""
        with self._async_lock:
            return self._async_frame

    def _capture_loop(self) -> None:
        while True:
            frame = self.capture(self._region)
            if frame is not None:
                with self._async_lock:
                    self._async_frame = frame
