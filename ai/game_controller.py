"""
GameController — understands 8 Ball Pool mobile UI and executes shots.

Game UI layout (scrcpy frame coordinates):
  ┌─────────────────────────────────────────────────────────┐
  │  [Power Bar]    [  TABLE  ]                  [Wheel]   │
  │   left side                                 right side  │
  └─────────────────────────────────────────────────────────┘

Shooting sequence:
  1. Drag the RIGHT WHEEL up/down → rotates aim angle
  2. Drag the LEFT POWER BAR back → charges power
  3. Release → fires the shot

The wheel position and power bar position are stored in:
  calibration/ui_scrcpy.json
Run `python ai/game_controller.py --calibrate` to set them interactively.
"""
import json
import logging
import math
import os
import sys
import time

# Allow running as `python ai/game_controller.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from ai.scrcpy_touch import ScrcpyTouch

log = logging.getLogger(__name__)

_UI_CFG = os.path.join(
    os.path.dirname(__file__), "..", "calibration", "ui_scrcpy.json")

# How many frame-pixels of wheel drag correspond to 1 degree of aim change.
# Tune this after testing — start at 4.0 and adjust.
DEFAULT_PX_PER_DEG = 4.0

# Power: drag the cue bar this many px back from its rest position
POWER_DRAG_PX = {
    "light":  200,
    "medium": 400,
    "hard":   620,
}


class GameController:

    def __init__(self, touch: ScrcpyTouch, frame_w: int, frame_h: int):
        self._touch  = touch
        self._fw     = frame_w
        self._fh     = frame_h
        self._cfg    = self._load_cfg()

    # ── public ────────────────────────────────────────────────────────────────

    def aim_and_shoot(self, cue_pos: tuple, ghost_pos: tuple,
                      current_ghost: tuple | None,
                      power: str = "medium") -> bool:
        """
        Full shot sequence:
          1. Rotate aim to ghost_pos using the wheel
          2. Charge and release power bar
        Returns True if executed.
        """
        if not self._cfg:
            log.error("UI not calibrated — run: python ai/game_controller.py --calibrate")
            return False

        # ── 1. Calculate angle difference ────────────────────────────────────
        target_angle = self._angle(cue_pos, ghost_pos)

        if current_ghost is not None:
            current_angle = self._angle(cue_pos, current_ghost)
        else:
            current_angle = target_angle   # assume already aimed correctly

        delta_deg = self._angle_diff(target_angle, current_angle)
        log.info("aim: target=%.1f°  current=%.1f°  delta=%.1f°",
                 math.degrees(target_angle),
                 math.degrees(current_angle),
                 math.degrees(delta_deg))

        # ── 2. Rotate wheel ───────────────────────────────────────────────────
        if abs(math.degrees(delta_deg)) > 1.0:
            self._rotate_wheel(delta_deg)
            time.sleep(0.15)   # let game catch up

        # ── 3. Shoot ──────────────────────────────────────────────────────────
        self._shoot(power)
        return True

    # ── wheel ─────────────────────────────────────────────────────────────────

    def _rotate_wheel(self, delta_rad: float) -> None:
        """
        Drag the right wheel to rotate the aim by delta_rad radians.
        Positive delta → drag UP (counterclockwise).
        Negative delta → drag DOWN (clockwise).
        """
        px_per_deg = self._cfg.get("px_per_deg", DEFAULT_PX_PER_DEG)
        drag_px    = int(math.degrees(abs(delta_rad)) * px_per_deg)
        drag_px    = max(5, min(drag_px, 400))

        wx = self._cfg["wheel_x"]
        wy = self._cfg["wheel_y"]

        if delta_rad > 0:
            end_y = wy - drag_px   # drag up
        else:
            end_y = wy + drag_px   # drag down

        end_y = max(10, min(self._fh - 10, end_y))
        log.info("wheel drag: (%d,%d) -> (%d,%d)  [%.0fpx]", wx, wy, wx, end_y, drag_px)
        self._touch.swipe(wx, wy, wx, end_y, duration_ms=300, steps=25)

    # ── power bar / shoot ─────────────────────────────────────────────────────

    def _shoot(self, power: str = "medium") -> None:
        """
        Drag the left power bar back by the given power level, then release.
        """
        drag = POWER_DRAG_PX.get(power, POWER_DRAG_PX["medium"])

        px = self._cfg["power_x"]
        py = self._cfg["power_y"]

        pull_y = py + drag   # pull DOWN to charge
        pull_y = min(self._fh - 10, pull_y)

        log.info("shoot: power=%s  drag=%dpx  (%d,%d)->(%d,%d)",
                 power, drag, px, py, px, pull_y)

        self._touch.swipe(px, py, px, pull_y, duration_ms=600, steps=30)

    # ── geometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def _angle(origin: tuple, target: tuple) -> float:
        """Angle in radians from origin to target (atan2)."""
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        return math.atan2(dy, dx)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed shortest difference between two angles (radians)."""
        d = a - b
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    # ── config ────────────────────────────────────────────────────────────────

    def _load_cfg(self) -> dict:
        if not os.path.exists(_UI_CFG):
            return {}
        try:
            with open(_UI_CFG) as f:
                raw = json.load(f)
        except Exception as e:
            log.warning("Could not load UI config: %s", e)
            return {}

        # Scale absolute coords to current frame size if calib was at different size
        saved_w = raw.get("frame_w", self._fw)
        saved_h = raw.get("frame_h", self._fh)
        sx = self._fw / saved_w
        sy = self._fh / saved_h
        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
            log.info("UI calib scaled %.2fx%.2f  (saved %dx%d → current %dx%d)",
                     sx, sy, saved_w, saved_h, self._fw, self._fh)
        cfg = dict(raw)
        cfg["wheel_x"] = int(raw["wheel_x"] * sx)
        cfg["wheel_y"] = int(raw["wheel_y"] * sy)
        cfg["power_x"] = int(raw["power_x"] * sx)
        cfg["power_y"] = int(raw["power_y"] * sy)
        return cfg

    @property
    def is_calibrated(self) -> bool:
        return bool(self._cfg)


# ── interactive calibration ───────────────────────────────────────────────────

def _capture_hwnd(hwnd: int, w: int, h: int):
    """
    Capture a window's contents using PrintWindow API — works even when
    the window is behind other windows or minimized.
    """
    import ctypes, win32ui, win32con
    import numpy as np
    try:
        hwnd_dc  = win32gui.GetWindowDC(hwnd)
        mfc_dc   = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc  = mfc_dc.CreateCompatibleDC()
        bmp      = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        # PW_RENDERFULLCONTENT = 2 — renders even GPU-composited content
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
        raw = bmp.GetBitmapBits(True)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        log.warning("PrintWindow failed (%s) — using blank frame", e)
        return 255 * np.ones((h, w, 3), dtype=np.uint8)


def calibrate_ui() -> None:
    """
    Click on the wheel center and the power bar center to calibrate.
    Uses ScreenCapture (same as calibrate.py) so the frame is correct.
    Saves to calibration/ui_scrcpy.json.
    """
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    from pipeline.capture import ScreenCapture
    import time as _time

    cap = ScreenCapture()
    print("Looking for scrcpy window...")
    region = cap.find_scrcpy_window()
    if region is None:
        print("ERROR: scrcpy window not found.")
        return

    cap.start_async(region)

    frame = None
    for _ in range(60):
        frame = cap.latest_frame()
        if frame is not None:
            break
        _time.sleep(0.05)

    if frame is None:
        print("ERROR: Could not capture frame.")
        return

    fh, fw = frame.shape[:2]
    scale = min(1.0, 1280 / fw, 800 / fh)
    dw, dh = int(fw * scale), int(fh * scale)

    cfg   = {}
    click = []

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((int(x / scale), int(y / scale)))

    win = "UI Calibration  (ESC=cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.setMouseCallback(win, _mouse)

    steps = [
        ("RIGHT WHEEL — click the CENTER of the aim wheel", "wheel"),
        ("LEFT POWER BAR — click the CENTER of the power/cue bar", "power"),
    ]

    for prompt, key in steps:
        click.clear()
        print(f"\n  {prompt}")
        while not click:
            live = cap.latest_frame()
            if live is not None:
                frame = live
            disp = cv2.resize(frame, (dw, dh))
            cv2.putText(disp, prompt, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            for k2 in ("wheel", "power"):
                if f"{k2}_x" in cfg:
                    dx = int(cfg[f"{k2}_x"] * scale)
                    dy = int(cfg[f"{k2}_y"] * scale)
                    cv2.circle(disp, (dx, dy), 14, (0, 255, 0), 3)
            cv2.imshow(win, disp)
            if cv2.waitKey(30) == 27:
                cv2.destroyAllWindows()
                return
        x, y = click[0]
        cfg[f"{key}_x"] = x
        cfg[f"{key}_y"] = y
        cfg["px_per_deg"] = DEFAULT_PX_PER_DEG
        print(f"  Set {key} = ({x}, {y})")

    cv2.destroyAllWindows()
    cfg["frame_w"] = fw
    cfg["frame_h"] = fh
    os.makedirs(os.path.dirname(_UI_CFG), exist_ok=True)
    with open(_UI_CFG, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved UI calibration to {_UI_CFG}")
    print(f"  wheel=({cfg['wheel_x']}, {cfg['wheel_y']})")
    print(f"  power=({cfg['power_x']}, {cfg['power_y']})")
    print(f"  px_per_deg={cfg['px_per_deg']}")
    print("\nTip: adjust 'px_per_deg' in the JSON if aim angle is off.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_ui()
