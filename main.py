"""
8 Ball Pool — AI Shot Assistant
================================
Entry point. Wires the pipeline stages together and runs the main loop.

Two concurrent loops
--------------------
  Fast loop  (~60 FPS) — capture → classify → track → shot → overlay
  YOLO loop  (GPU rate) — runs AsyncYOLOInference in its own thread

The fast loop never blocks on the GPU. It submits each frame to the async
YOLO worker and immediately reads back the last completed result. There is a
one-YOLO-cycle lag (~55 ms) in detections, but the EWA tracker absorbs it
transparently — the overlay stays smooth at 60 FPS regardless of GPU speed.

DPI awareness MUST be set before any other Windows API call — so it lives
at the very top of this file, before any imports.
"""
# ── DPI awareness (must be first) ─────────────────────────────────────────────
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)    # Per-monitor v1
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()     # Fallback for older Windows
    except Exception:
        pass

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import cv2
import numpy as np

from config import TABLE_RECALC, TARGET_FPS, CUE_WHITENESS_THRESH, SKIP_TOP_FRAC_SCRCPY, SCRCPY_EXPAND_PX
from models import Shot, TableBounds
from overlay.window import OverlayWindow
from pipeline.capture import ScreenCapture
from pipeline.classifier import classify_ball, whiteness_score
from pipeline.inference import AsyncYOLOInference
from pipeline.shot_engine import best_physics_shot, shot_from_ghost
from pipeline.table_detector import TableDetector
from pipeline.tracker import BallTracker, GhostBuffer


class ShotAssistant:
    """
    Orchestrator — owns the pipeline components and coordinates them.
    Holds no domain logic itself; all logic lives in the pipeline modules.
    """

    def __init__(self):
        self._capture    = ScreenCapture()
        self._table_det  = None   # created in start() once source is known
        self._yolo       = AsyncYOLOInference()   # GPU runs in its own thread
        self._tracker    = BallTracker()
        self._ghost_buf  = GhostBuffer()
        self._overlay:   Optional[OverlayWindow] = None

        self._table:          Optional[TableBounds] = None
        self._table_age:      int  = 0
        self._table_locked:   bool = False   # True when calibration is loaded (no auto re-detect)
        self._table_lock      = threading.Lock()
        self._region:         Optional[tuple] = None
        self._canvas_offset:  tuple[float, float] = (0.0, 0.0)
        self._my_type:        Optional[str] = None

    # ── startup ───────────────────────────────────────────────────────────────

    def start(self, source: str = "auto") -> None:
        source = _resolve_source(source)

        if source == "chrome":
            self._region = self._capture.find_game_window()
            if self._region is None:
                log.error("Chrome window not found. Open Chrome at 8ballpool.com/game")
                sys.exit(1)
            self._table_det = TableDetector()
        else:  # scrcpy
            self._region = self._capture.find_scrcpy_window()
            if self._region is None:
                log.error("scrcpy window not found. Start scrcpy first.")
                sys.exit(1)
            self._table_det = TableDetector(skip_top_frac=SKIP_TOP_FRAC_SCRCPY,
                                            expand_px=SCRCPY_EXPAND_PX)

        # Load saved calibration (overrides auto-detection if present)
        calib = _load_calibration(source)
        if calib:
            with self._table_lock:
                self._table = calib
            self._table_locked = True
            log.info("Calibration loaded: %s  r=%d", calib.as_tuple(), calib.ball_radius)

        x1, y1, x2, y2 = self._region
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            log.error("Invalid window size %dx%d. Restore the Chrome window and try again.", w, h)
            sys.exit(1)

        # Clamp origin to screen edge; offset corrects coordinate translation
        sx, sy = max(0, x1), max(0, y1)
        self._canvas_offset = (float(sx - x1), float(sy - y1))

        log.info("Region=%s  (%dx%d)", self._region, w, h)
        log.info("Canvas origin=(%d,%d)  offset=%s", sx, sy, self._canvas_offset)

        self._overlay = OverlayWindow(sx, sy, w, h)
        self._overlay.on_type_change = self._set_my_type
        self._overlay.setup()

        # Start capture thread before the pipeline — the pipeline reads
        # latest_frame() so it never blocks on PrintWindow.
        self._capture.start_async(self._region)

        pipeline = threading.Thread(target=self._run_pipeline, daemon=True)
        pipeline.start()

        self._overlay.run()   # blocks until the overlay window is closed

    def _set_my_type(self, t: str) -> None:
        self._my_type = t

    # ── main pipeline loop ────────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        interval    = 1.0 / TARGET_FPS
        fps_buf:    list[float] = []
        debug_saved = False
        raw_frame_saved = False   # save one raw frame immediately for diagnosis

        while True:
            t0 = time.perf_counter()

            # ── 1. Capture (reads latest frame from async capture thread) ────
            frame = self._capture.latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # Save a raw frame once so we can verify capture is working
            if not raw_frame_saved:
                cv2.imwrite("debug_raw.png", frame)
                log.info("debug_raw.png saved — shape=%s", frame.shape)
                raw_frame_saved = True

            # ── 2. Table detection (offloaded to background thread) ───────────
            # Skipped when calibration is loaded — manual bounds never change.
            self._table_age += 1
            if not self._table_locked and (self._table is None or self._table_age >= TABLE_RECALC):
                self._table_age = 0
                _frame_copy = frame.copy()
                threading.Thread(
                    target=self._detect_table_bg,
                    args=(_frame_copy,), daemon=True,
                ).start()

            with self._table_lock:
                table_snap = self._table

            if table_snap is None:
                time.sleep(0.05)
                continue

            pockets = table_snap.pockets()

            # ── 3. YOLO inference (non-blocking) ─────────────────────────────
            # submit() queues the frame for the GPU thread and returns instantly.
            # latest_result() returns whatever the GPU finished last cycle.
            # The 1-frame lag is invisible: EWA tracking smooths it out.
            roi = frame[table_snap.y1:table_snap.y2, table_snap.x1:table_snap.x2]
            self._yolo.submit(roi, offset=(table_snap.x1, table_snap.y1))
            raw_balls, raw_colls = self._yolo.latest_result()

            if not raw_balls:
                # Save annotated frame (table only) so we can diagnose YOLO misses
                if not debug_saved:
                    self._save_debug(frame, [], table_snap.pockets(), table_snap)
                    debug_saved = True
                self._overlay.push_shot(None)
                self._log_fps(fps_buf, t0, n_balls=0, has_shot=False)
                time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
                continue

            # ── 4. Classify each detection ────────────────────────────────────
            for b in raw_balls:
                b["ws"] = whiteness_score(b["patch"], b["r"])

            cue_idx = max(range(len(raw_balls)), key=lambda i: raw_balls[i]["ws"])
            has_cue = raw_balls[cue_idx]["ws"] > CUE_WHITENESS_THRESH

            for i, b in enumerate(raw_balls):
                if i == cue_idx and has_cue:
                    b["type"] = "cue"; b["subtype"] = "cue"
                else:
                    b["type"], b["subtype"] = classify_ball(b["patch"], b["r"])

            # ── 5. Update ball-radius estimate (slow EWA — very stable) ───────
            measured = max(7, min(30, int(np.median([b["r"] for b in raw_balls]))))
            R = table_snap.ball_radius = int(
                round(table_snap.ball_radius * 0.85 + measured * 0.15))

            # ── 6. Track + smooth positions ───────────────────────────────────
            tracks = self._tracker.update(raw_balls)
            balls  = [
                {
                    "pos":     (int(round(tr["pos"][0])), int(round(tr["pos"][1]))),
                    "radius":  tr["r"],
                    "type":    tr["type"],
                    "subtype": tr["subtype"],
                }
                for tr in tracks
            ]

            cue_pos = next((b["pos"] for b in balls if b["type"] == "cue"), None)
            if cue_pos is None:
                self._overlay.push_shot(None)
                self._log_fps(fps_buf, t0, len(balls), has_shot=False)
                time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
                continue

            # ── 7. Stabilise ghost ball ───────────────────────────────────────
            ghost_pos = self._ghost_buf.push(raw_colls)

            # ── 8. Select best shot ───────────────────────────────────────────
            # PATH A: player is aiming — use YOLO ghost ball
            shot: Optional[Shot] = (
                shot_from_ghost(
                    cue_pos, ghost_pos, self._ghost_buf.aim_conf,
                    balls, pockets, table_snap, R,
                )
                if ghost_pos else None
            )
            # PATH B: player is not aiming — suggest best physics shot
            if shot is None:
                shot = best_physics_shot(
                    cue_pos, balls, self._my_type, pockets, table_snap, R)

            # ── 9. Translate frame coords → overlay canvas coords ─────────────
            if shot:
                dx, dy = self._canvas_offset
                shot = shot.translate(dx, dy)

            # ── Debug frame saved once on startup ─────────────────────────────
            if not debug_saved and raw_balls:
                self._save_debug(frame, balls, pockets, table_snap)
                debug_saved = True

            self._overlay.push_shot(shot)
            self._log_fps(fps_buf, t0, len(balls), has_shot=shot is not None)
            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _detect_table_bg(self, frame: np.ndarray) -> None:
        """Run table detection in a background thread and update self._table."""
        detected = self._table_det.detect(frame)
        if detected:
            with self._table_lock:
                self._table = detected
            log.info("Table=%s  r=%d", detected.as_tuple(), detected.ball_radius)

    def _log_fps(self, buf: list, t0: float, n_balls: int, has_shot: bool) -> None:
        buf.append(time.perf_counter() - t0)
        if len(buf) > TARGET_FPS:
            buf.pop(0)
            fps = len(buf) / sum(buf)
            print(f"FPS:{fps:5.1f}  balls:{n_balls:2d}  shot:{'yes' if has_shot else 'no '}",
                  end="\r")

    def _save_debug(self, frame: np.ndarray, balls: list,
                    pockets: list, table: TableBounds) -> None:
        """Save an annotated frame to debug_frame.png once on startup."""
        dbg = frame.copy()
        if table:
            x1, y1, x2, y2 = table.as_tuple()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(dbg, f"TABLE r={table.ball_radius}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i, (px, py) in enumerate(pockets):
            cv2.circle(dbg, (int(px), int(py)), 20, (0, 0, 255), 3)
            cv2.putText(dbg, str(i), (int(px) - 8, int(py) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for b in balls:
            col = (0, 255, 0) if b["type"] == "cue" else (0, 165, 255)
            cv2.circle(dbg, b["pos"], b["radius"], col, 2)
            cv2.putText(dbg, b["type"][:3],
                        (b["pos"][0], b["pos"][1] - b["radius"] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imwrite("debug_frame.png", dbg)
        log.info("debug_frame.png saved — balls=%d  r=%d", len(balls), table.ball_radius)


# ── source selection ──────────────────────────────────────────────────────────

def _resolve_source(source: str) -> str:
    """Return 'chrome' or 'scrcpy'. Prompts the user if source=='auto'."""
    if source in ("chrome", "scrcpy"):
        return source
    print("\nSelect capture source:")
    print("  1) Chrome  (اللعبة في المتصفح)")
    print("  2) scrcpy  (الموبايل عبر USB/WiFi)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "chrome"
        if choice == "2":
            return "scrcpy"
        print("Please enter 1 or 2.")


# ── calibration loader ────────────────────────────────────────────────────────

def _load_calibration(source: str) -> "Optional[TableBounds]":
    path = os.path.join("calibration", f"table_{source}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            b = json.load(f)
        tw = b["x2"] - b["x1"]; th = b["y2"] - b["y1"]
        ball_r = max(7, min(30, max(int(tw / 96), int(th / 48))))
        return TableBounds(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                           ball_radius=ball_r)
    except Exception as e:
        log.warning("Could not load calibration: %s", e)
        return None


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8 Ball Pool AI Shot Assistant")
    parser.add_argument(
        "--source", choices=["chrome", "scrcpy"], default="auto",
        help="Capture source: 'chrome' for browser, 'scrcpy' for mobile mirror",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Open calibration tool to manually draw the table rectangle",
    )
    args = parser.parse_args()

    if args.calibrate:
        from calibrate import run_calibration
        source = _resolve_source(args.source)
        run_calibration(source)
    else:
        ShotAssistant().start(source=args.source)
