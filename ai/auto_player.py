"""
AutoPlayer — executes shots via scrcpy window touch injection (no ADB).

Uses Win32 PostMessage to send mouse events to the scrcpy SDL window.
scrcpy forwards them as touch events to the phone automatically.
"""
import logging
import threading
import time
import win32gui

from ai.scrcpy_touch import ScrcpyTouch
from ai.game_controller import GameController, calibrate_ui

log = logging.getLogger(__name__)

POST_SHOT_WAIT = 4.5    # seconds to wait after shot for animation


class AutoPlayer:

    def __init__(self, hwnd: int, frame_w: int, frame_h: int,
                 title_bar_h: int = 30):
        self._hwnd    = hwnd
        self._fw      = frame_w
        self._fh      = frame_h
        self._tbar    = title_bar_h

        self._touch   = ScrcpyTouch(hwnd, title_bar_h)
        self._ctrl    = GameController(self._touch, frame_w, frame_h)

        if not self._ctrl.is_calibrated:
            log.warning("UI not calibrated — run: "
                        "python ai/game_controller.py --calibrate")

    @property
    def ready(self) -> bool:
        return self._ctrl.is_calibrated

    def execute_shot(self, cue_pos: tuple, ghost_pos: tuple,
                     current_ghost: tuple | None = None,
                     power: str = "medium") -> bool:
        """
        Aim at ghost_pos and fire.
        current_ghost: the ghost ball currently detected (current aim direction).
        """
        log.info("AutoPlayer: executing shot  ghost=(%.0f,%.0f)  power=%s",
                 *ghost_pos, power)

        ok = self._ctrl.aim_and_shoot(
            cue_pos, ghost_pos, current_ghost, power)

        if ok:
            log.info("AutoPlayer: shot sent — waiting %.1fs", POST_SHOT_WAIT)
            time.sleep(POST_SHOT_WAIT)
        return ok


# ── calibration entry point ───────────────────────────────────────────────────

def find_scrcpy_hwnd() -> int | None:
    result = []
    def _cb(h, _):
        if win32gui.IsWindowVisible(h) and win32gui.GetClassName(h) == "SDL_app":
            result.append(h)
    win32gui.EnumWindows(_cb, None)
    return result[0] if result else None


if __name__ == "__main__":
    hwnd = find_scrcpy_hwnd()
    if not hwnd:
        print("scrcpy window not found — start scrcpy first.")
    else:
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        print(f"scrcpy window found: hwnd={hwnd}  size={w}x{h}")
        calibrate_ui(hwnd, w, h)
