"""
8 Ball Pool – AI Shot Assistant
================================
DPI awareness must be set BEFORE any other Windows API call.
"""
# ── DPI awareness (must be first) ─────────────────────────────────────────────
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import sys
import threading
import time

import cv2
import numpy as np

from capture  import ScreenCapture
from detector import BallDetector
from overlay  import TransparentOverlay
from physics  import best_shot


TARGET_FPS   = 60
DEBUG_WINDOW = False   # set True to open a live OpenCV debug view
SAVE_DEBUG   = True    # saves debug_frame.png on first frame so you can inspect it


class EightBallAssistant:
    def __init__(self):
        self.cap         = ScreenCapture()
        self.det         = BallDetector()
        self.overlay     = None
        self.region      = None
        self._dx         = 0
        self._dy         = 0
        self._running    = False
        self._table_cache     = None   # cached table bounds
        self._table_frame_ctr = 0      # frames since last table re-detect

    def start(self):
        self.region = self.cap.find_game_window()
        if self.region is None:
            print("\n[ERROR] Chrome window not found.\n"
                  "  Open Chrome at 8ballpool.com/game\n")
            sys.exit(1)

        x1, y1, x2, y2 = self.region
        w, h = x2 - x1, y2 - y1

        sx = max(0, x1)
        sy = max(0, y1)
        self._dx = sx - x1   # frame → canvas x offset (usually 11 or 7)
        self._dy = sy - y1   # frame → canvas y offset (usually 0)

        print(f"[Main] Capture region : {self.region}  ({w}×{h})")
        print(f"[Main] Overlay origin : ({sx},{sy})  "
              f"frame→canvas offset: ({self._dx},{self._dy})")

        self.overlay = TransparentOverlay(sx, sy, w, h)
        self.overlay.on_type_change = self.det.set_my_type
        self.overlay.setup()

        self._running = True
        t = threading.Thread(target=self._process_loop, daemon=True)
        t.start()

        self.overlay.run()
        self._running = False
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()

    # ── processing thread ─────────────────────────────────────────────────────

    def _process_loop(self):
        interval    = 1.0 / TARGET_FPS
        stats_buf   = []
        saved_debug = False

        while self._running:
            t0 = time.perf_counter()

            # ── capture ───────────────────────────────────────────────────────
            frame = self.cap.capture(self.region)
            if frame is None:
                time.sleep(0.05)
                continue

            # ── Step 1: detect table (cached — re-detect every 90 frames) ─────
            self._table_frame_ctr += 1
            if self._table_cache is None or self._table_frame_ctr >= 90:
                table = self.det.detect_table(frame)
                if table:
                    self._table_cache = table
                    self._table_frame_ctr = 0
                    print(f"\n[Main] Table={table}  r={self.det.ball_radius}")
            else:
                table = self._table_cache
                # Keep detector's table_bounds in sync
                self.det.table_bounds = table

            # ── Step 2: detect pockets (from table bounds) ────────────────────
            pockets = self.det.detect_pockets(table) if table else []

            # ── Steps 3-5: YOLO detects balls + ghost ball → shot ────────────
            balls, shot = self.det.detect_shot(frame)

            # ── translate frame coords → canvas coords ────────────────────────
            if shot:
                shot = self._to_canvas(shot)

            self.overlay.push_shot(shot)

            # ── save annotated debug frame (once, after detection) ────────────
            if SAVE_DEBUG and not saved_debug:
                dbg = frame.copy()
                if table:
                    x1t,y1t,x2t,y2t = table
                    cv2.rectangle(dbg,(x1t,y1t),(x2t,y2t),(0,255,0),3)
                    cv2.putText(dbg,f"TABLE r={self.det.ball_radius}",
                                (x1t,max(0,y1t-10)),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                for i,(px,py) in enumerate(pockets):
                    cv2.circle(dbg,(px,py),20,(0,0,255),3)
                    cv2.putText(dbg,str(i),(px-8,py+6),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                for b in balls:
                    col=(0,255,0) if b["type"]=="cue" else (0,165,255)
                    cv2.circle(dbg,b["pos"],b["radius"],col,2)
                    cv2.putText(dbg,b["type"][:3],(b["pos"][0],b["pos"][1]-b["radius"]-2),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
                cv2.imwrite("debug_frame.png", dbg)
                print(f"[Main] Saved debug_frame.png  "
                      f"table={table}  balls={len(balls)}  r={self.det.ball_radius}")
                saved_debug = True

            # ── debug window ──────────────────────────────────────────────────
            if DEBUG_WINDOW:
                self._show_debug(frame, table, balls, shot)

            # ── FPS ───────────────────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            stats_buf.append(elapsed)
            if len(stats_buf) > TARGET_FPS:
                stats_buf.pop(0)
                fps = len(stats_buf) / sum(stats_buf)
                print(
                    f"[Main] FPS:{fps:5.1f}  balls:{len(balls):2d}  "
                    f"table:{'yes' if table else 'no '}  "
                    f"shot:{'yes' if shot else 'no '}",
                    end="\r"
                )

            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_canvas(self, shot: dict) -> dict:
        """Translate all positions from frame coords to overlay canvas coords."""
        dx, dy = self._dx, self._dy

        def adj(pos):
            return (pos[0] - dx, pos[1] - dy) if pos else None

        return {
            **shot,
            "cue_pos":    adj(shot.get("cue_pos")),
            "target_pos": adj(shot.get("target_pos")),
            "ghost_pos":  adj(shot.get("ghost_pos")),
            "pocket":     adj(shot.get("pocket")),
            "cue_path":   [adj(p) for p in shot.get("cue_path", [])],
        }

    def _show_debug(self, frame, table, balls, shot):
        dbg = frame.copy()
        if table:
            x1, y1, x2, y2 = table
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 80, 0), 3)
        for b in balls:
            col = (0, 255, 0) if b["type"] == "cue" else (0, 140, 255)
            cv2.circle(dbg, b["pos"], b["radius"], col, 2)
            cv2.putText(dbg, b["type"][:3], b["pos"],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        # Scale for display
        h, w = dbg.shape[:2]
        sc   = min(1.0, 1400 / w)
        if sc < 1.0:
            dbg = cv2.resize(dbg, (int(w * sc), int(h * sc)))
        cv2.imshow("8BP Debug", dbg)
        cv2.waitKey(1)


if __name__ == "__main__":
    # Quick debug helper: python main.py --debug
    if "--debug" in sys.argv:
        DEBUG_WINDOW = True
    if "--save" in sys.argv:
        SAVE_DEBUG = True

    app = EightBallAssistant()
    app.start()
