"""
Temporal tracking for detected balls.

Two classes, two responsibilities:

  BallTracker  — nearest-neighbour matching + EWA position smoothing.
                 Eliminates per-frame jitter caused by YOLO's bounding-box
                 variance even when the ball isn't moving.

  GhostBuffer  — stabilises ghost-ball (collision class) detections across
                 frames via a rolling median. One missed frame won't kill
                 the aiming overlay.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from config import (
    GHOST_BUF_LEN, GHOST_MIN_CONF, GHOST_STALE,
    TRACK_ALPHA, TRACK_EXPIRE, TRACK_MAX_DIST,
)

Vec2 = tuple[float, float]


def _dist(a: Vec2, b: Vec2) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ── Ball tracker ──────────────────────────────────────────────────────────────

class BallTracker:
    """
    Maintains a dict of live tracks, each representing one ball on the table.

    Matching strategy: nearest-neighbour with a max-distance threshold.
    Smoothing strategy: EWA with alpha = TRACK_ALPHA.
      alpha = 0.35 means "35% new detection, 65% history" — fast enough to
      follow a moving ball, slow enough to absorb single-frame jitter.

    A track is dropped after TRACK_EXPIRE frames without a matching detection,
    which handles balls being pocketed or temporarily occluded.
    """

    def __init__(self):
        self._tracks:  dict[int, dict] = {}
        self._next_id: int             = 0
        self._frame_n: int             = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Match detections to tracks, smooth positions, expire stale tracks.
        Returns the list of all active track dicts with smoothed 'pos'.
        """
        self._frame_n += 1
        used: set[int] = set()

        for det in detections:
            rx, ry = det["abs_xy"]

            # Find the closest unmatched track within the max-distance threshold
            best_id, best_d = None, TRACK_MAX_DIST
            for tid, tr in self._tracks.items():
                if tid in used:
                    continue
                d = _dist(tr["pos"], (rx, ry))
                if d < best_d:
                    best_d = d
                    best_id = tid

            a = TRACK_ALPHA
            if best_id is not None:
                tr = self._tracks[best_id]
                tr["pos"]     = (tr["pos"][0] * (1 - a) + rx * a,
                                 tr["pos"][1] * (1 - a) + ry * a)
                tr["r"]       = int(tr["r"] * 0.8 + det["r"] * 0.2)
                tr["type"]    = det["type"]
                tr["subtype"] = det["subtype"]
                tr["ws"]      = det.get("ws", 0.0)
                tr["patch"]   = det["patch"]
                tr["last"]    = self._frame_n
                used.add(best_id)
            else:
                # New ball appeared on the table (or first frame)
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {
                    "pos":     (float(rx), float(ry)),
                    "r":       det["r"],
                    "type":    det["type"],
                    "subtype": det["subtype"],
                    "ws":      det.get("ws", 0.0),
                    "patch":   det["patch"],
                    "last":    self._frame_n,
                }
                used.add(tid)

        # Expire tracks that haven't been matched recently
        stale = [tid for tid, tr in self._tracks.items()
                 if self._frame_n - tr["last"] > TRACK_EXPIRE]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())


# ── Ghost ball buffer ─────────────────────────────────────────────────────────

class GhostBuffer:
    """
    Smooths ghost-ball (YOLO class 1) detections over time.

    Uses a short rolling buffer of high-confidence positions and returns
    their component-wise median. This is robust to:
      - Single-frame misses (the buffer still has older positions)
      - Single-frame outliers (median is not affected by one bad point)

    aim_conf is exposed so the main loop can decide whether to show the
    AI-mode overlay or fall back to physics suggestions.
    """

    def __init__(self):
        self._buf:        deque         = deque(maxlen=GHOST_BUF_LEN)
        self._last_frame: int           = -999
        self._frame_n:    int           = 0
        self.aim_conf:    float         = 0.0

    def push(self, raw_colls: list[dict]) -> Vec2 | None:
        """
        Feed raw collision detections for the current frame.
        Returns the stable median ghost position, or None if the player
        is not aiming (buffer is empty or stale).
        """
        self._frame_n += 1

        if raw_colls:
            best          = max(raw_colls, key=lambda c: c["conf"])
            self.aim_conf = best["conf"]
            if best["conf"] >= GHOST_MIN_CONF:
                self._buf.append(best["abs_xy"])
                self._last_frame = self._frame_n
        else:
            self.aim_conf = 0.0

        # Clear buffer when the player has stopped aiming for a while
        if self._frame_n - self._last_frame > GHOST_STALE:
            self._buf.clear()

        if not self._buf:
            return None

        return (
            int(np.median([g[0] for g in self._buf])),
            int(np.median([g[1] for g in self._buf])),
        )
