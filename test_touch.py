"""
Quick test: rotate the aim wheel then fire the power bar.
Run with: python test_touch.py
"""
import json
import time
import cv2
import numpy as np
import win32api
import win32gui
import mss

from ai.scrcpy_touch import ScrcpyTouch


def screenshot_with_cursor(name: str, sx: int, sy: int) -> None:
    """Take a screenshot of the scrcpy window and mark the cursor position."""
    with mss.mss() as sct:
        mon = {"left": rect[0], "top": rect[1],
               "width": cur_w, "height": cur_h}
        raw = sct.grab(mon)
    img = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

    # cursor pos relative to window
    cx = sx - rect[0]
    cy = sy - rect[1]

    cv2.circle(img, (cx, cy), 30, (0, 0, 255), 3)
    cv2.circle(img, (cx, cy),  5, (0, 0, 255), -1)
    cv2.line(img, (cx - 40, cy), (cx + 40, cy), (0, 0, 255), 2)
    cv2.line(img, (cx, cy - 40), (cx, cy + 40), (0, 0, 255), 2)
    cv2.putText(img, name, (cx + 35, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    path = f"test_{name}.png"
    cv2.imwrite(path, img)
    print(f"    saved: {path}")

# ── find scrcpy window ────────────────────────────────────────────────────────

def find_scrcpy():
    result = []
    def _cb(h, _):
        if win32gui.IsWindowVisible(h) and win32gui.GetClassName(h) == "SDL_app":
            result.append(h)
    win32gui.EnumWindows(_cb, None)
    return result[0] if result else None

hwnd = find_scrcpy()
if not hwnd:
    print("ERROR: scrcpy window not found.")
    exit(1)

rect = win32gui.GetWindowRect(hwnd)
cur_w = rect[2] - rect[0]
cur_h = rect[3] - rect[1]
print(f"scrcpy: hwnd={hwnd}  rect={rect}  size={cur_w}x{cur_h}")

# ── load + scale calibration ──────────────────────────────────────────────────

with open("calibration/ui_scrcpy.json") as f:
    cfg = json.load(f)

saved_w = cfg.get("frame_w", cur_w)
saved_h = cfg.get("frame_h", cur_h)
scale_x = cur_w / saved_w
scale_y = cur_h / saved_h

wheel_x = int(cfg["wheel_x"] * scale_x)
wheel_y = int(cfg["wheel_y"] * scale_y)
power_x = int(cfg["power_x"] * scale_x)
power_y = int(cfg["power_y"] * scale_y)

print(f"scale={scale_x:.2f}x{scale_y:.2f}")
print(f"wheel frame=({wheel_x},{wheel_y})  →  screen=({rect[0]+wheel_x},{rect[1]+wheel_y})")
print(f"power frame=({power_x},{power_y})  →  screen=({rect[0]+power_x},{rect[1]+power_y})")

# ── screen bounds check ───────────────────────────────────────────────────────

sm_w = win32api.GetSystemMetrics(0)
sm_h = win32api.GetSystemMetrics(1)
print(f"monitor={sm_w}x{sm_h}")

def check(name, fx, fy):
    sx, sy = rect[0] + fx, rect[1] + fy
    ok = (0 <= sx < sm_w) and (0 <= sy < sm_h)
    print(f"  {name}: screen=({sx},{sy})  {'OK' if ok else '*** OUT OF SCREEN ***'}")

check("wheel", wheel_x, wheel_y)
check("power", power_x, power_y)

touch = ScrcpyTouch(hwnd, title_bar_h=30)

# ── TEST 1: move cursor to wheel and show actual position ─────────────────────
print("\n[1] Moving cursor to WHEEL ...")
win32gui.SetForegroundWindow(hwnd)
time.sleep(0.15)
win32api.SetCursorPos((rect[0] + wheel_x, rect[1] + wheel_y))
time.sleep(0.3)
actual = win32api.GetCursorPos()
print(f"    cursor now at: {actual}")
screenshot_with_cursor("wheel", *actual)

# ── TEST 2: move cursor to power bar and show actual position ─────────────────
print("[2] Moving cursor to POWER BAR ...")
win32api.SetCursorPos((rect[0] + power_x, rect[1] + power_y))
time.sleep(0.3)
actual = win32api.GetCursorPos()
print(f"    cursor now at: {actual}")
screenshot_with_cursor("power", *actual)

input("\nPress ENTER to perform the swipes...")

# ── TEST 3: wheel swipe ───────────────────────────────────────────────────────
print("[3] Wheel swipe UP ...")
touch.swipe(wheel_x, wheel_y, wheel_x, wheel_y - 100, duration_ms=500, steps=20)
time.sleep(0.5)

# ── TEST 4: power bar drag ────────────────────────────────────────────────────
print("[4] Power bar drag DOWN — FULL POWER (550px) ...")
touch.swipe(power_x, power_y, power_x, power_y + 620, duration_ms=1000, steps=60)

print("\nDone.")
