"""
8 Ball Pool — AI Shot Assistant
================================
Entry point. Wires the pipeline stages together and runs the main loop.

Pipeline (one iteration per frame, ~60 FPS):
  1. Capture     — grab Chrome window content via Win32 PrintWindow
  2. Table       — detect felt bounds (cached every TABLE_RECALC frames)
  3. Inference   — run YOLO on the table ROI
  4. Classify    — label each ball (cue / 8ball / solid / stripe)
  5. Track       — match detections to tracks, smooth with EWA
  6. Ghost       — stabilise ghost-ball position across frames
  7. Shot        — select best shot (AI path or physics fallback)
  8. Render      — push shot to overlay for drawing

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

import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import TABLE_RECALC, TARGET_FPS
from models import Shot, TableBounds
from overlay.window import OverlayWindow
from pipeline.capture import ScreenCapture
from pipeline.classifier import classify_ball, whiteness_score
from pipeline.inference import YOLOInference
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
        self._table_det  = TableDetector()
        self._yolo       = YOLOInference()
        self._tracker    = BallTracker()
        self._ghost_buf  = GhostBuffer()
        self._overlay:   Optional[OverlayWindow] = None

        self._table:          Optional[TableBounds] = None
        self._table_age:      int  = 0      # frames since last table re-detection
        self._region:         Optional[tuple] = None   # screen capture region
        self._canvas_offset:  tuple[float, float] = (0.0, 0.0)
        self._my_type:        Optional[str] = None     # "solid" | "stripe" | None

    # ── startup ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._region = self._capture.find_game_window()
        if self._region is None:
            print("\n[ERROR] Chrome window not found.\n"
                  "  Open Chrome at 8ballpool.com/game\n")
            sys.exit(1)

        x1, y1, x2, y2 = self._region
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            print(f"\n[ERROR] Invalid window size {w}×{h}.\n"
                  f"  Restore the Chrome window and try again.\n")
            sys.exit(1)

        # Clamp origin to screen edge; offset corrects coordinate translation
        sx, sy = max(0, x1), max(0, y1)
        self._canvas_offset = (float(sx - x1), float(sy - y1))

        print(f"[Main] Region={self._region}  ({w}×{h})")
        print(f"[Main] Canvas origin=({sx},{sy})  offset={self._canvas_offset}")

        self._overlay = OverlayWindow(sx, sy, w, h)
        self._overlay.on_type_change = self._set_my_type
        self._overlay.setup()

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

        while True:
            t0 = time.perf_counter()

            # ── 1. Capture ────────────────────────────────────────────────────
            frame = self._capture.capture(self._region)
            if frame is None:
                time.sleep(0.05)
                continue

            # ── 2. Table detection (cached; full re-detect every N frames) ────
            self._table_age += 1
            if self._table is None or self._table_age >= TABLE_RECALC:
                detected = self._table_det.detect(frame)
                if detected:
                    self._table     = detected
                    self._table_age = 0
                    print(f"\n[Main] Table={self._table.as_tuple()}  "
                          f"r={self._table.ball_radius}")

            if self._table is None:
                time.sleep(0.05)
                continue

            pockets = self._table.pockets()

            # ── 3. YOLO inference ─────────────────────────────────────────────
            t = self._table
            roi = frame[t.y1:t.y2, t.x1:t.x2]
            raw_balls, raw_colls = self._yolo.run(roi, offset=(t.x1, t.y1))

            if not raw_balls:
                self._overlay.push_shot(None)
                self._log_fps(fps_buf, t0, n_balls=0, has_shot=False)
                time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
                continue

            # ── 4. Classify each detection ────────────────────────────────────
            for b in raw_balls:
                b["ws"] = whiteness_score(b["patch"], b["r"])

            cue_idx = max(range(len(raw_balls)), key=lambda i: raw_balls[i]["ws"])
            has_cue = raw_balls[cue_idx]["ws"] > 0.22

            for i, b in enumerate(raw_balls):
                if i == cue_idx and has_cue:
                    b["type"] = "cue"; b["subtype"] = "cue"
                else:
                    b["type"], b["subtype"] = classify_ball(b["patch"], b["r"])

            # ── 5. Update ball-radius estimate (slow EWA — very stable) ───────
            measured = max(7, min(30, int(np.median([b["r"] for b in raw_balls]))))
            R = self._table.ball_radius = int(
                round(self._table.ball_radius * 0.85 + measured * 0.15))

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
                    balls, pockets, self._table, R,
                )
                if ghost_pos else None
            )
            # PATH B: player is not aiming — suggest best physics shot
            if shot is None:
                shot = best_physics_shot(
                    cue_pos, balls, self._my_type, pockets, self._table, R)

            # ── 9. Translate frame coords → overlay canvas coords ─────────────
            if shot:
                dx, dy = self._canvas_offset
                shot = shot.translate(dx, dy)

            # ── Debug frame saved once on startup ─────────────────────────────
            if not debug_saved:
                self._save_debug(frame, balls, pockets)
                debug_saved = True

            self._overlay.push_shot(shot)
            self._log_fps(fps_buf, t0, len(balls), has_shot=shot is not None)
            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log_fps(self, buf: list, t0: float, n_balls: int, has_shot: bool) -> None:
        buf.append(time.perf_counter() - t0)
        if len(buf) > TARGET_FPS:
            buf.pop(0)
            fps = len(buf) / sum(buf)
            print(
                f"[Main] FPS:{fps:5.1f}  balls:{n_balls:2d}  "
                f"shot:{'yes' if has_shot else 'no '}",
                end="\r",
            )

    def _save_debug(self, frame: np.ndarray, balls: list, pockets: list) -> None:
        """Save an annotated frame to debug_frame.png once on startup."""
        dbg = frame.copy()
        if self._table:
            x1, y1, x2, y2 = self._table.as_tuple()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(dbg, f"TABLE r={self._table.ball_radius}",
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
        print(f"[Main] debug_frame.png saved — "
              f"balls={len(balls)}  r={self._table.ball_radius if self._table else '?'}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ShotAssistant().start()
