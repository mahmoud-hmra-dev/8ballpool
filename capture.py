"""
Screen capture using Win32 PrintWindow API.

PrintWindow captures the window's OWN CONTENT directly from its DC,
regardless of whether the window is behind other applications.
This is essential: the user may have VS Code, terminal, or other windows
on top of Chrome while the game is running.

PW_RENDERFULLCONTENT = 0x2  →  forces Chrome (GPU-accelerated) to blit.
Falls back to mss if PrintWindow returns 0.
"""
import threading
import numpy as np
import cv2
import win32gui
import win32ui
from ctypes import windll
import mss as _mss_module


class ScreenCapture:
    def __init__(self):
        self._region   = None
        self._hwnd     = None    # Chrome HWND
        self._win_rect = None    # Full window rect (incl. shadow border)
        self._local    = threading.local()   # per-thread mss (fallback)
        # Cached GDI objects — created once, reused every frame
        self._save_dc  = None
        self._bmp      = None
        self._win_w    = 0
        self._win_h    = 0

    # ── find window ───────────────────────────────────────────────────────────

    def find_game_window(self) -> tuple | None:
        found = []

        def _cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if "8 Ball Pool" in title:
                found.append((10, hwnd, win32gui.GetWindowRect(hwnd)))
            elif "Chrome" in title or "chrome" in title:
                found.append((1,  hwnd, win32gui.GetWindowRect(hwnd)))

        win32gui.EnumWindows(_cb, None)
        if not found:
            return None

        found.sort(reverse=True)
        _, hwnd, rect = found[0]
        x1, y1, x2, y2 = rect
        print(f"[Capture] Window: '{win32gui.GetWindowText(hwnd)}'  rect={rect}")

        self._hwnd     = hwnd
        self._win_rect = rect

        # Skip browser chrome (~90 px: title+tabs+address bar)
        y1 += 90
        self._region = (x1, y1, x2, y2)
        return self._region

    # ── GDI cache ─────────────────────────────────────────────────────────────

    def _init_dc(self):
        """Create save_dc + bitmap once; reused every frame (saves ~10 Win32 calls/frame)."""
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
        Capture the Chrome window content using PrintWindow.
        Works even when Chrome is behind VS Code or any other window.
        GDI objects are created once and reused every frame.
        """
        r = region or self._region
        if r is None or self._hwnd is None:
            return None

        if self._save_dc is None:
            self._init_dc()

        x1, y1, x2, y2 = r
        wx1, wy1 = self._win_rect[0], self._win_rect[1]

        try:
            # PW_RENDERFULLCONTENT = 0x2  →  works for GPU-accelerated Chrome
            ok = windll.user32.PrintWindow(self._hwnd, self._save_dc.GetSafeHdc(), 2)

            if not ok:
                print("[Capture] PrintWindow returned 0, falling back to mss")
                return self._mss_capture(r)

            raw  = self._bmp.GetBitmapBits(True)
            full = np.frombuffer(raw, dtype='uint8').reshape(
                self._win_h, self._win_w, 4)

            # Crop: convert screen coords → window-local coords
            cx1 = max(0, x1 - wx1);  cy1 = max(0, y1 - wy1)
            cx2 = min(self._win_w, x2 - wx1)
            cy2 = min(self._win_h, y2 - wy1)

            cropped = full[cy1:cy2, cx1:cx2].copy()   # .copy() for crop safety
            return cv2.cvtColor(cropped, cv2.COLOR_BGRA2BGR)

        except Exception as e:
            print(f"[Capture] PrintWindow error: {e}")
            self._save_dc = None   # force re-init next frame
            return self._mss_capture(r)

    # ── mss fallback ──────────────────────────────────────────────────────────

    def _mss_capture(self, region: tuple) -> np.ndarray | None:
        if getattr(self._local, "sct", None) is None:
            self._local.sct = _mss_module.mss()
        x1, y1, x2, y2 = region
        mon = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
        try:
            raw  = self._local.sct.grab(mon)
            bgra = np.array(raw)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"[Capture] mss error: {e}")
            self._local.sct = None
            return None
