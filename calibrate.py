"""
Table calibration tool.

Usage:
  python calibrate.py --source scrcpy
  python calibrate.py --source chrome

Drag a rectangle over the table, then press ENTER to save.
The saved calibration is loaded automatically by main.py on next run.
"""
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import argparse
import json
import os
import time

import cv2
import numpy as np

from pipeline.capture import ScreenCapture

CALIB_DIR = "calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

# ── mouse state ───────────────────────────────────────────────────────────────

_s = {"drag": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}


def _mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _s.update(drag=True, x0=x, y0=y, x1=x, y1=y)
    elif event == cv2.EVENT_MOUSEMOVE and _s["drag"]:
        _s["x1"] = x; _s["y1"] = y
    elif event == cv2.EVENT_LBUTTONUP:
        _s["drag"] = False; _s["x1"] = x; _s["y1"] = y


# ── main calibration loop ─────────────────────────────────────────────────────

def run_calibration(source: str) -> bool:
    cap = ScreenCapture()

    print(f"\nLooking for {source} window...")
    region = cap.find_scrcpy_window() if source == "scrcpy" else cap.find_game_window()
    if region is None:
        print(f"ERROR: {source} window not found.")
        return False

    cap.start_async(region)

    frame = None
    for _ in range(60):
        frame = cap.latest_frame()
        if frame is not None:
            break
        time.sleep(0.05)

    if frame is None:
        print("ERROR: Could not capture frame.")
        return False

    fh, fw = frame.shape[:2]
    # Scale display to fit inside 1280×800
    scale = min(1.0, 1280 / fw, 800 / fh)
    dw, dh = int(fw * scale), int(fh * scale)

    # Pre-load existing calibration
    save_path = os.path.join(CALIB_DIR, f"table_{source}.json")
    if os.path.exists(save_path):
        with open(save_path) as f:
            prev = json.load(f)
        _s.update(
            x0=int(prev["x1"] * scale), y0=int(prev["y1"] * scale),
            x1=int(prev["x2"] * scale), y1=int(prev["y2"] * scale),
        )
        print(f"Existing calibration loaded from {save_path}")

    win = f"Calibrate Table [{source}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.setMouseCallback(win, _mouse)

    print("\n" + "─" * 50)
    print("  Drag  : draw rectangle over the table")
    print("  ENTER : save and exit")
    print("  R     : reset rectangle")
    print("  ESC   : cancel without saving")
    print("─" * 50 + "\n")

    while True:
        live = cap.latest_frame()
        if live is not None:
            frame = live

        disp = cv2.resize(frame, (dw, dh))

        # Draw current rectangle
        rx1 = min(_s["x0"], _s["x1"]); ry1 = min(_s["y0"], _s["y1"])
        rx2 = max(_s["x0"], _s["x1"]); ry2 = max(_s["y0"], _s["y1"])
        has_rect = (rx2 - rx1 > 10) and (ry2 - ry1 > 10)

        if has_rect:
            cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            # Corner handles
            for cx, cy in [(rx1, ry1), (rx2, ry1), (rx1, ry2), (rx2, ry2)]:
                cv2.circle(disp, (cx, cy), 5, (0, 255, 0), -1)
            # Dimensions label
            real_w = int((rx2 - rx1) / scale)
            real_h = int((ry2 - ry1) / scale)
            real_x1 = int(rx1 / scale); real_y1 = int(ry1 / scale)
            label = f"  ({real_x1}, {real_y1})  {real_w} x {real_h} px  "
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = max(0, rx1)
            ly = max(th + 4, ry1 - 4)
            cv2.rectangle(disp, (lx, ly - th - 4), (lx + tw, ly + 2), (0, 0, 0), -1)
            cv2.putText(disp, label, (lx, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Instructions overlay
        hints = [
            "Drag to draw table rectangle",
            "ENTER = save    R = reset    ESC = cancel",
        ]
        for i, txt in enumerate(hints):
            cv2.putText(disp, txt, (8, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if not has_rect:
            cv2.putText(disp, "No rectangle drawn yet", (8, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # Enter
            if not has_rect:
                print("Draw a rectangle first.")
                continue
            bounds = {
                "x1": int(rx1 / scale), "y1": int(ry1 / scale),
                "x2": int(rx2 / scale), "y2": int(ry2 / scale),
                "source": source,
            }
            with open(save_path, "w") as f:
                json.dump(bounds, f, indent=2)
            print(f"Saved: {bounds}")
            print(f"File : {save_path}")
            cv2.destroyAllWindows()
            return True

        elif key in (ord('r'), ord('R')):
            _s.update(x0=0, y0=0, x1=0, y1=0)

        elif key == 27:  # ESC
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return False


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table calibration tool")
    parser.add_argument("--source", choices=["chrome", "scrcpy"], required=True,
                        help="Capture source")
    args = parser.parse_args()
    run_calibration(args.source)
