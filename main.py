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

from ai.data_collector import DataCollector
from ai.predictor import ShotPredictor
from ai.auto_player import AutoPlayer
from config import TABLE_RECALC, TARGET_FPS, CUE_WHITENESS_THRESH, SKIP_TOP_FRAC_SCRCPY, SCRCPY_EXPAND_PX
from models import Shot, TableBounds
from overlay.window import OverlayWindow
from pipeline.capture import ScreenCapture
from pipeline.classifier import classify_ball, whiteness_score
from pipeline.game_state import GameStateTracker
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
        self._table_locked:   bool = False
        self._table_lock      = threading.Lock()
        self._region:         Optional[tuple] = None
        self._canvas_offset:  tuple[float, float] = (0.0, 0.0)
        self._my_type:        Optional[str] = None
        self._collector:      Optional[DataCollector] = None
        self._predictor:      Optional[ShotPredictor] = None
        self._game_state:     GameStateTracker = GameStateTracker()
        self._auto_player:    Optional[AutoPlayer]    = None
        self._autoplay_state: str = "IDLE"   # IDLE | WAITING
        self._last_shot_time: float = 0.0
        # guided mode
        self._guided:            bool  = False
        self._guided_state:      str   = "IDLE"   # IDLE | WAITING
        self._fire_requested:    bool  = False
        # self-play
        self._selfplay:          bool  = False
        self._pre_shot_record:   Optional[dict] = None
        self._pre_shot_n_balls:  int   = 0
        self._selfplay_success:  int   = 0
        self._selfplay_total:    int   = 0
        self._retrain_proc              = None   # subprocess
        self._RETRAIN_EVERY:     int   = 20     # retrain after N successful shots

    # ── startup ───────────────────────────────────────────────────────────────

    def start(self, source: str = "auto", collect: bool = False,
              autoplay: bool = False, selfplay: bool = False,
              guided: bool = False) -> None:
        source = _resolve_source(source)
        if collect:
            self._collector = DataCollector()
        if autoplay or selfplay or guided:
            self._predictor = ShotPredictor()
        if selfplay:
            self._selfplay = True
        if guided:
            self._guided = True
            print("Guided mode: aim manually, press F8 to fire the shot.")

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

        if (autoplay or guided) and self._predictor and self._predictor.ready:
            self._auto_player = AutoPlayer(
                hwnd=self._capture._hwnd, frame_w=w, frame_h=h)

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
                    t, sub = classify_ball(b["patch"], b["r"])
                    # Ghost ball (aim indicator) can look white — don't call it cue
                    if t == "cue":
                        t, sub = "ball", "ball"
                    b["type"] = t; b["subtype"] = sub

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

            # ── guided: poll F8 key ───────────────────────────────────────────
            if self._guided and not self._fire_requested:
                import keyboard
                if keyboard.is_pressed("f8"):
                    self._fire_requested = True
                    log.info("Guided: F8 pressed — will fire")

            # ── 7b. Game state tracking ──────────────────────────────────────
            self._game_state.update(balls, self._my_type or "unknown")

            # ── 7d. Data collection ───────────────────────────────────────────
            if self._collector is not None:
                self._collector.push(
                    balls, cue_pos, ghost_pos,
                    self._ghost_buf.aim_conf,
                    self._my_type, table_snap,
                )

            # ── 8. Select best shot ───────────────────────────────────────────
            if self._guided and self._auto_player is not None:
                shot = self._run_guided(balls, cue_pos, ghost_pos, pockets,
                                        table_snap, R)
            elif self._auto_player is not None:
                # AUTO-PLAY: model predicts ghost, ADB fires the shot
                shot = self._run_autoplay(balls, cue_pos, ghost_pos, pockets,
                                          table_snap, R)
            else:
                # MANUAL: follow player aim or suggest physics shot
                shot = (
                    shot_from_ghost(
                        cue_pos, ghost_pos, self._ghost_buf.aim_conf,
                        balls, pockets, table_snap, R,
                    )
                    if ghost_pos else None
                )
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
            _extra = f"state:{self._autoplay_state}  cue:{'ok' if cue_pos else 'NO'}  ghost:{'yes' if ghost_pos else 'no'}" if self._auto_player else ""
            self._log_fps(fps_buf, t0, len(balls), has_shot=shot is not None, extra=_extra)
            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _detect_table_bg(self, frame: np.ndarray) -> None:
        """Run table detection in a background thread and update self._table."""
        detected = self._table_det.detect(frame)
        if detected:
            with self._table_lock:
                self._table = detected
            log.info("Table=%s  r=%d", detected.as_tuple(), detected.ball_radius)

    def _run_autoplay(self, balls, cue_pos, ghost_pos, pockets,
                      table_snap, R) -> "Optional[Shot]":
        """
        Auto-play state machine:
          IDLE    → ghost visible (our turn) → predict & fire → WAITING
          WAITING → wait POST_SHOT_WAIT seconds → IDLE
        """
        import threading

        now = time.time()
        POST_SHOT_WAIT = 5.5  # seconds to wait after firing

        # ── WAITING: shot just fired, wait for animation ──────────────────────
        if self._autoplay_state == "WAITING":
            if now - self._last_shot_time > POST_SHOT_WAIT:
                # ── check retrain finished → reload model ─────────────────────
                if self._retrain_proc and self._retrain_proc.poll() == 0:
                    log.info("Self-play: retrain done — reloading model")
                    self._predictor = ShotPredictor()
                    self._retrain_proc = None

                # ── evaluate shot outcome ──────────────────────────────────────
                if self._selfplay and self._pre_shot_record:
                    n_now = len([b for b in balls if b["type"] != "cue"])
                    scored = n_now < self._pre_shot_n_balls
                    self._selfplay_total += 1
                    if scored:
                        self._selfplay_success += 1
                        self._save_selfplay_shot()
                        log.info("Self-play: SCORED  success=%d/%d",
                                 self._selfplay_success, self._selfplay_total)
                        if self._selfplay_success % self._RETRAIN_EVERY == 0:
                            self._trigger_retrain()
                    else:
                        log.info("Self-play: missed  success=%d/%d",
                                 self._selfplay_success, self._selfplay_total)
                    self._pre_shot_record = None

                self._autoplay_state = "IDLE"
                log.info("AutoPlay: ready for next shot")

            if ghost_pos is not None:
                return shot_from_ghost(cue_pos, ghost_pos,
                                       self._ghost_buf.aim_conf,
                                       balls, pockets, table_snap, R)
            return None

        # ── IDLE: fire when ghost is visible (= our turn to aim) ─────────────
        if self._autoplay_state == "IDLE" and ghost_pos is not None:
            ai_ghost = self._pick_target(
                balls, cue_pos, pockets, table_snap, R)

            if ai_ghost:
                log.info("AutoPlay: firing toward (%.0f, %.0f)  current_ghost=(%.0f, %.0f)",
                         *ai_ghost, *ghost_pos)
                self._autoplay_state = "WAITING"
                self._last_shot_time = now

                if self._selfplay:
                    tw = table_snap.x2 - table_snap.x1
                    th = table_snap.y2 - table_snap.y1
                    self._pre_shot_record = {
                        "ts":       now,
                        "my_type":  self._my_type or "unknown",
                        "cue_n":    [round((cue_pos[0]-table_snap.x1)/tw, 4),
                                     round((cue_pos[1]-table_snap.y1)/th, 4)],
                        "ghost_n":  [round((ai_ghost[0]-table_snap.x1)/tw, 4),
                                     round((ai_ghost[1]-table_snap.y1)/th, 4)],
                        "balls": [
                            {"pos_n": [round((b["pos"][0]-table_snap.x1)/tw, 4),
                                       round((b["pos"][1]-table_snap.y1)/th, 4)],
                             "type":    b["type"],
                             "subtype": b.get("subtype", "")}
                            for b in balls if b["type"] != "cue"
                        ],
                    }
                    self._pre_shot_n_balls = len(
                        [b for b in balls if b["type"] != "cue"])

                threading.Thread(
                    target=self._auto_player.execute_shot,
                    args=(cue_pos, ai_ghost, ghost_pos),
                    daemon=True,
                ).start()
                return shot_from_ghost(
                    cue_pos, ai_ghost, 1.0,
                    balls, pockets, table_snap, R)

        # Fallback: show current aim or physics suggestion
        if ghost_pos is not None:
            return shot_from_ghost(cue_pos, ghost_pos,
                                   self._ghost_buf.aim_conf,
                                   balls, pockets, table_snap, R)
        return best_physics_shot(cue_pos, balls, self._my_type,
                                 pockets, table_snap, R)

    def _request_fire(self) -> None:
        """Called when user presses F8."""
        self._fire_requested = True

    def _run_guided(self, balls, cue_pos, ghost_pos, pockets,
                    table_snap, R) -> "Optional[Shot]":
        """
        Phase 1 — Guided mode:
          Aim manually with the game wheel.
          Press F8 → AI fires at the current ghost position and records the shot.
        """
        now = time.time()
        POST = 5.5

        # ── WAITING: shot fired, wait for animation ───────────────────────────
        if self._guided_state == "WAITING":
            if now - self._last_shot_time > POST:
                if self._pre_shot_record is not None:
                    n_now = len([b for b in balls if b["type"] != "cue"])
                    if n_now < self._pre_shot_n_balls:
                        self._selfplay_success += 1
                        self._save_selfplay_shot()
                        log.info("Guided: SCORED  success=%d/%d",
                                 self._selfplay_success, self._selfplay_total)
                        if self._selfplay_success % self._RETRAIN_EVERY == 0:
                            self._trigger_retrain()
                    else:
                        log.info("Guided: missed  success=%d/%d",
                                 self._selfplay_success, self._selfplay_total)
                    self._selfplay_total += 1
                    self._pre_shot_record = None
                self._guided_state  = "IDLE"
                self._fire_requested = False
            if ghost_pos:
                return shot_from_ghost(cue_pos, ghost_pos,
                                       self._ghost_buf.aim_conf,
                                       balls, pockets, table_snap, R)
            return None

        # ── IDLE: waiting for F8 press ────────────────────────────────────────
        if not self._fire_requested or ghost_pos is None:
            if ghost_pos:
                return shot_from_ghost(cue_pos, ghost_pos,
                                       self._ghost_buf.aim_conf,
                                       balls, pockets, table_snap, R)
            return best_physics_shot(cue_pos, balls, self._my_type,
                                     pockets, table_snap, R)

        # ── F8 pressed → fire! ────────────────────────────────────────────────
        self._fire_requested = False
        target = ghost_pos
        log.info("Guided: FIRING at (%.0f, %.0f)", *target)
        self._guided_state   = "WAITING"
        self._last_shot_time = now

        tw = table_snap.x2 - table_snap.x1
        th = table_snap.y2 - table_snap.y1
        self._pre_shot_record = {
            "ts":      now,
            "my_type": self._my_type or "unknown",
            "cue_n":   [round((cue_pos[0]-table_snap.x1)/tw, 4),
                        round((cue_pos[1]-table_snap.y1)/th, 4)],
            "ghost_n": [round((target[0]-table_snap.x1)/tw, 4),
                        round((target[1]-table_snap.y1)/th, 4)],
            "balls": [
                {"pos_n": [round((b["pos"][0]-table_snap.x1)/tw, 4),
                            round((b["pos"][1]-table_snap.y1)/th, 4)],
                 "type":    b["type"],
                 "subtype": b.get("subtype", "")}
                for b in balls if b["type"] != "cue"
            ],
        }
        self._pre_shot_n_balls = len([b for b in balls if b["type"] != "cue"])

        threading.Thread(
            target=self._auto_player.execute_shot,
            args=(cue_pos, target, ghost_pos),
            daemon=True,
        ).start()
        return shot_from_ghost(cue_pos, target, 1.0,
                               balls, pockets, table_snap, R)

    def _pick_target(self, balls, cue_pos, pockets, table_snap, R):
        """
        Choose where to aim:
          - Self-play: physics best shot + random exploration noise
          - Auto-play: ML model prediction (falls back to physics)
        """
        import math, random

        # Physics gives us the geometrically best shot
        physics = best_physics_shot(cue_pos, balls, self._my_type,
                                    pockets, table_snap, R)
        physics_ghost = physics.ghost_pos if physics else None

        if self._selfplay and physics_ghost:
            # Add random angle noise for exploration (±12°, biased toward ±0°)
            noise_deg = random.gauss(0, 6)   # gaussian: mostly small, occasionally bigger
            noise_deg = max(-15, min(15, noise_deg))  # clamp
            if noise_deg != 0:
                dx = physics_ghost[0] - cue_pos[0]
                dy = physics_ghost[1] - cue_pos[1]
                angle = math.atan2(dy, dx) + math.radians(noise_deg)
                dist  = math.hypot(dx, dy)
                physics_ghost = (
                    cue_pos[0] + dist * math.cos(angle),
                    cue_pos[1] + dist * math.sin(angle),
                )
            log.info("Self-play target: physics+noise(%.1f°)=(%.0f,%.0f)",
                     noise_deg, *physics_ghost)
            return physics_ghost

        # Auto-play: try ML model first, fall back to physics
        if self._predictor:
            ml_ghost = self._predictor.predict(
                balls, cue_pos, self._my_type, table_snap)
            if ml_ghost:
                return ml_ghost

        return physics_ghost

    def _save_selfplay_shot(self) -> None:
        """Save the last successful self-play shot to the dataset."""
        import json as _json
        from datetime import datetime
        rec = self._pre_shot_record
        if rec is None:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        path  = os.path.join("ai", "dataset", f"selfplay_{today}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(_json.dumps(rec) + "\n")

    def _trigger_retrain(self) -> None:
        """Launch ai/train.py in a background subprocess."""
        if self._retrain_proc and self._retrain_proc.poll() is None:
            log.info("Self-play: retrain already running — skipping")
            return
        import subprocess
        log.info("Self-play: launching retrain (success=%d)...", self._selfplay_success)
        self._retrain_proc = subprocess.Popen(
            [sys.executable, os.path.join("ai", "train.py")],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

    def _log_fps(self, buf: list, t0: float, n_balls: int, has_shot: bool,
                 extra: str = "") -> None:
        buf.append(time.perf_counter() - t0)
        if len(buf) > TARGET_FPS:
            buf.pop(0)
            fps = len(buf) / sum(buf)
            print(f"FPS:{fps:5.1f}  balls:{n_balls:2d}  shot:{'yes' if has_shot else 'no '}  {extra}",
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
    parser.add_argument(
        "--collect", action="store_true",
        help="Record your shots to ai/dataset/ for AI training",
    )
    parser.add_argument(
        "--autoplay", action="store_true",
        help="AI plays automatically using the trained model (requires shot_model.pt)",
    )
    parser.add_argument(
        "--selfplay", action="store_true",
        help="AI plays and learns from successful shots — retrains every 20 scored balls",
    )
    parser.add_argument(
        "--guided", action="store_true",
        help="Phase 1: you aim manually, AI fires automatically and records successful shots",
    )
    args = parser.parse_args()

    if args.calibrate:
        from calibrate import run_calibration
        source = _resolve_source(args.source)
        run_calibration(source)
    else:
        ShotAssistant().start(source=args.source,
                              collect=args.collect,
                              autoplay=args.autoplay or args.selfplay,
                              selfplay=args.selfplay,
                              guided=args.guided)
