"""
Data Collector — records aiming shots while you play.

Each record captures:
  - Ball positions (normalized to table size)
  - Cue ball position
  - Ghost ball position (where the player aimed)
  - Player type (solid / stripe)

Records are saved as JSON Lines to ai/dataset/shots_YYYY-MM-DD.jsonl
One line = one shot.

State machine prevents duplicate recordings of the same shot:
  IDLE → AIMING (ghost appears) → RECORDED → IDLE (ghost disappears)
"""

import json
import os
import time
from datetime import date
from typing import Optional

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# Minimum ghost displacement (normalized) to consider it a new shot
_NEW_SHOT_THRESH = 0.04   # 4% of table width/height

# Minimum aim confidence before we record
_MIN_CONF = 0.55


class DataCollector:
    """
    Call push() every pipeline frame.
    Records one entry per stable aiming position.
    """

    def __init__(self):
        today = date.today().isoformat()
        self._path = os.path.join(DATASET_DIR, f"shots_{today}.jsonl")
        self._f    = open(self._path, "a", encoding="utf-8")

        self._state          = "IDLE"    # IDLE | AIMING | RECORDED
        self._last_ghost_n   = None      # last recorded ghost (normalized)
        self._stable_frames  = 0
        self._total          = self._count_existing()

        print(f"[DataCollector] saving to {self._path}  (existing={self._total})")

    # ── public API ────────────────────────────────────────────────────────────

    def push(self,
             balls:     list,
             cue_pos:   tuple,
             ghost_pos: Optional[tuple],
             aim_conf:  float,
             my_type:   Optional[str],
             table) -> None:
        """
        Call every frame. Records a shot when the ghost ball is stable.

        Parameters
        ----------
        balls      : list of dicts with keys 'pos', 'type', 'subtype'
        cue_pos    : (x, y) in frame coords
        ghost_pos  : (x, y) or None
        aim_conf   : float 0–1 from GhostBuffer
        my_type    : "solid" | "stripe" | None
        table      : TableBounds
        """
        if ghost_pos is None or aim_conf < _MIN_CONF:
            # Ghost gone → reset so next appearance is a new shot
            if self._state == "AIMING":
                self._state = "IDLE"
                self._stable_frames = 0
            elif self._state == "RECORDED":
                self._state = "IDLE"
            return

        # Normalize everything to [0, 1] relative to table
        tw = table.x2 - table.x1
        th = table.y2 - table.y1
        if tw <= 0 or th <= 0:
            return

        def _n(pos):
            return [
                round((pos[0] - table.x1) / tw, 4),
                round((pos[1] - table.y1) / th, 4),
            ]

        ghost_n = _n(ghost_pos)

        if self._state == "IDLE":
            self._state       = "AIMING"
            self._stable_frames = 1

        elif self._state == "AIMING":
            self._stable_frames += 1
            # Record after 8 stable frames to avoid transient detections
            if self._stable_frames >= 8:
                # Skip if same as last recorded shot (player didn't re-aim)
                if self._last_ghost_n is not None:
                    dx = abs(ghost_n[0] - self._last_ghost_n[0])
                    dy = abs(ghost_n[1] - self._last_ghost_n[1])
                    if dx < _NEW_SHOT_THRESH and dy < _NEW_SHOT_THRESH:
                        self._state = "RECORDED"
                        return

                self._record(balls, cue_pos, ghost_n, my_type, table, tw, th)
                self._last_ghost_n = ghost_n
                self._state        = "RECORDED"

        # RECORDED → wait for ghost to disappear (handled at top)

    def close(self):
        self._f.close()

    @property
    def total(self) -> int:
        return self._total

    # ── internals ─────────────────────────────────────────────────────────────

    def _record(self, balls, cue_pos, ghost_n, my_type, table, tw, th):
        def _nb(pos):
            return [round((pos[0]-table.x1)/tw, 4),
                    round((pos[1]-table.y1)/th, 4)]

        record = {
            "ts":       round(time.time(), 3),
            "my_type":  my_type or "unknown",
            "cue_n":    _nb(cue_pos),
            "ghost_n":  ghost_n,
            "balls": [
                {
                    "pos_n":   _nb(b["pos"]),
                    "type":    b["type"],
                    "subtype": b.get("subtype", ""),
                }
                for b in balls
                if b["type"] != "cue"
            ],
        }

        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        self._total += 1
        print(f"\r[DataCollector] shots recorded: {self._total}   ", end="", flush=True)

    def _count_existing(self) -> int:
        if not os.path.exists(self._path):
            return 0
        with open(self._path, encoding="utf-8") as f:
            return sum(1 for _ in f)
