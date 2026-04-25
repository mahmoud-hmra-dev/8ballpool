"""
Auto-calibrate px_per_deg for the aim wheel.

Steps:
  1. Make sure scrcpy is open and the game is in AIM mode (aim line visible)
  2. Run:  python calibrate_wheel.py
  3. Click the CUE BALL (white ball)
  4. Click the END of the AIM LINE (tip of the dotted line)
  5. Script rotates wheel by a known amount
  6. Click the NEW END of the aim line
  7. px_per_deg is calculated and saved automatically

Repeat 3+ times to get a more accurate average.
"""
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import json
import math
import os
import time

import cv2

from ai.scrcpy_touch import ScrcpyTouch
from pipeline.capture import ScreenCapture

# ── constants ─────────────────────────────────────────────────────────────────

DRAG_PX    = 250          # how many px to rotate the wheel (UP)
N_TRIALS   = 3            # how many measurements to average
UI_CFG     = os.path.join("calibration", "ui_scrcpy.json")

# ── helpers ───────────────────────────────────────────────────────────────────

def angle_deg(origin, target):
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))


def angle_diff_deg(a, b):
    d = a - b
    while d >  180: d -= 360
    while d < -180: d += 360
    return d


def show_frame(win, frame, scale, prompt, points, colors):
    disp = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
    cv2.putText(disp, prompt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(disp, "ESC = cancel", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    for (px, py), color in zip(points, colors):
        dpx, dpy = int(px * scale), int(py * scale)
        cv2.circle(disp, (dpx, dpy), 10, color, 2)
        cv2.circle(disp, (dpx, dpy),  3, color, -1)
    cv2.imshow(win, disp)

# ── main ──────────────────────────────────────────────────────────────────────

def run():
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
        time.sleep(0.05)

    if frame is None:
        print("ERROR: Could not capture frame.")
        return

    if not os.path.exists(UI_CFG):
        print(f"ERROR: {UI_CFG} not found. Run `python ai/game_controller.py --calibrate` first.")
        return

    with open(UI_CFG) as f:
        cfg = json.load(f)

    fh, fw = frame.shape[:2]
    saved_w = cfg.get("frame_w", fw)
    saved_h = cfg.get("frame_h", fh)
    sx, sy  = fw / saved_w, fh / saved_h

    wheel_x = int(cfg["wheel_x"] * sx)
    wheel_y = int(cfg["wheel_y"] * sy)

    hwnd  = cap._hwnd
    touch = ScrcpyTouch(hwnd, title_bar_h=30)

    scale = min(1.0, 1280 / fw, 800 / fh)
    dw, dh = int(fw * scale), int(fh * scale)

    WIN = "Wheel Calibration"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh)

    measurements = []
    print(f"\nWill run {N_TRIALS} measurements and average them.")
    print(f"Rotating wheel UP by {DRAG_PX}px each trial.\n")

    for trial in range(1, N_TRIALS + 1):
        print(f"─── Trial {trial}/{N_TRIALS} ───")
        clicks = []

        # get latest live frame
        live = cap.latest_frame()
        if live is not None:
            frame = live

        def _mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((int(x / scale), int(y / scale)))

        cv2.setMouseCallback(WIN, _mouse)

        # ── Step 1: click cue ball ────────────────────────────────────────────
        print("  [1] Click the CUE BALL (white ball)")
        while len(clicks) < 1:
            live = cap.latest_frame()
            if live is not None:
                frame = live
            show_frame(WIN, frame, scale,
                       "1) Click the CUE BALL", clicks[:1], [(0, 255, 255)])
            if cv2.waitKey(30) == 27:
                cv2.destroyAllWindows(); return

        cue_pos = clicks[0]
        print(f"      cue = {cue_pos}")

        # ── Step 2: click aim line tip ────────────────────────────────────────
        print("  [2] Click the END of the AIM LINE (dotted line tip)")
        while len(clicks) < 2:
            live = cap.latest_frame()
            if live is not None:
                frame = live
            show_frame(WIN, frame, scale,
                       "2) Click the AIM LINE tip",
                       clicks[:2], [(0,255,255),(0,255,0)])
            if cv2.waitKey(30) == 27:
                cv2.destroyAllWindows(); return

        aim_before = clicks[1]
        angle_before = angle_deg(cue_pos, aim_before)
        print(f"      aim_before = {aim_before}  angle = {angle_before:.1f}°")

        # ── Step 3: rotate wheel ──────────────────────────────────────────────
        print(f"  [3] Rotating wheel UP {DRAG_PX}px ...")
        touch.swipe(wheel_x, wheel_y,
                    wheel_x, wheel_y - DRAG_PX,
                    duration_ms=600, steps=30)
        time.sleep(1.0)   # let game settle

        # ── Step 4: re-capture frame ──────────────────────────────────────────
        clicks.clear()
        live = cap.latest_frame()
        if live is not None:
            frame = live

        # ── Step 5: click new aim line tip ────────────────────────────────────
        print("  [4] Click the NEW aim line tip (after rotation)")
        while len(clicks) < 1:
            live = cap.latest_frame()
            if live is not None:
                frame = live
            show_frame(WIN, frame, scale,
                       "4) Click NEW AIM LINE tip",
                       [], [])
            if cv2.waitKey(30) == 27:
                cv2.destroyAllWindows(); return

        aim_after  = clicks[0]
        angle_after = angle_deg(cue_pos, aim_after)
        print(f"      aim_after  = {aim_after}  angle = {angle_after:.1f}°")

        # ── Calculate ─────────────────────────────────────────────────────────
        delta = abs(angle_diff_deg(angle_after, angle_before))
        if delta < 0.5:
            print("      WARNING: angle change too small — skipping this trial")
        else:
            px_per_deg = DRAG_PX / delta
            measurements.append(px_per_deg)
            print(f"      Δangle = {delta:.1f}°  →  px_per_deg = {px_per_deg:.2f}")

        # Rotate back to original position
        print("  Rotating back to original ...")
        touch.swipe(wheel_x, wheel_y,
                    wheel_x, wheel_y + DRAG_PX,
                    duration_ms=600, steps=30)
        time.sleep(1.0)
        print()

    cv2.destroyAllWindows()

    if not measurements:
        print("No valid measurements collected.")
        return

    avg = sum(measurements) / len(measurements)
    print(f"\n{'─'*40}")
    print(f"Results: {[f'{m:.2f}' for m in measurements]}")
    print(f"Average px_per_deg = {avg:.3f}")
    print(f"Old     px_per_deg = {cfg.get('px_per_deg', 4.0):.3f}")

    cfg["px_per_deg"] = round(avg, 3)
    with open(UI_CFG, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved to {UI_CFG}")


if __name__ == "__main__":
    run()
