"""
Game state tracker.

Watches ball counts each frame and detects:
  - Scratch      (cue ball potted)
  - Foul         (8-ball potted while my balls remain)
  - Win          (8-ball potted after clearing my balls)
  - Balls potted (solid / stripe disappear)

Writes game_state.json every ~0.5 s for external monitoring.
"""
from __future__ import annotations

import json
import time
from collections import deque

GAME_STATE_PATH = "game_state.json"
_SMOOTH = 8   # frames to average counts over (absorbs YOLO misses)


class GameStateTracker:

    def __init__(self) -> None:
        self._my_type:    str   = "unknown"
        self._history:    deque = deque(maxlen=_SMOOTH)
        self._stable:     dict  = {"solid": 7, "stripe": 7, "8ball": 1, "cue": 1}
        self._prev:       dict  = dict(self._stable)
        self._events:     list  = []
        self._last_write: float = 0.0

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, balls: list, my_type: str) -> dict:
        self._my_type = my_type

        # Count what YOLO sees this frame
        raw = {"solid": 0, "stripe": 0, "8ball": 0, "cue": 0}
        for b in balls:
            t = b.get("type", "")
            sub = b.get("subtype", t)
            if t == "cue":
                raw["cue"] += 1
            elif t == "8ball":
                raw["8ball"] += 1
            elif sub == "solid":
                raw["solid"] += 1
            elif sub == "stripe":
                raw["stripe"] += 1

        self._history.append(raw)

        # Smooth: round(mean) — stable against single-frame misses
        if len(self._history) >= 3:
            smooth = {
                k: int(round(sum(h[k] for h in self._history) / len(self._history)))
                for k in raw
            }
            self._detect_events(smooth)
            self._stable = smooth

        state = self._build_state()
        now = time.time()
        if now - self._last_write >= 0.5:
            self._write(state)
            self._last_write = now

        return state

    # ── internals ─────────────────────────────────────────────────────────────

    def _opp_type(self) -> str:
        if self._my_type == "solid":  return "stripe"
        if self._my_type == "stripe": return "solid"
        return "unknown"

    def _detect_events(self, new: dict) -> None:
        prev = self._prev

        # Scratch — cue ball disappeared
        if prev.get("cue", 1) >= 1 and new.get("cue", 0) == 0:
            self._event("scratch", "Cue ball potted — FOUL (scratch)")

        # 8-ball disappeared
        if prev.get("8ball", 1) >= 1 and new.get("8ball", 0) == 0:
            my_left = new.get(self._my_type, 1) if self._my_type in ("solid", "stripe") else 1
            if my_left > 0:
                self._event("foul_8ball", "8-ball potted early — FOUL (loss)")
            else:
                self._event("win", "8-ball potted after clearing — WIN")

        # Solid balls reduced
        diff_solid = prev.get("solid", 0) - new.get("solid", 0)
        if diff_solid > 0:
            who = "mine" if self._my_type == "solid" else "opponent"
            self._event("potted_solid",
                        f"{diff_solid} solid ball(s) potted [{who}]")

        # Stripe balls reduced
        diff_stripe = prev.get("stripe", 0) - new.get("stripe", 0)
        if diff_stripe > 0:
            who = "mine" if self._my_type == "stripe" else "opponent"
            self._event("potted_stripe",
                        f"{diff_stripe} stripe ball(s) potted [{who}]")

        self._prev = dict(new)

    def _event(self, etype: str, msg: str) -> None:
        entry = {"type": etype, "message": msg, "time": round(time.time(), 2)}
        self._events.append(entry)
        self._events = self._events[-20:]   # keep last 20
        print(f"[GameState] {msg}")

    def _build_state(self) -> dict:
        opp = self._opp_type()
        my_left  = self._stable.get(self._my_type, "?") \
                   if self._my_type in ("solid", "stripe") else "?"
        opp_left = self._stable.get(opp, "?") \
                   if opp != "unknown" else "?"

        return {
            "timestamp":               round(time.time(), 2),
            "my_type":                 self._my_type,
            "my_balls_remaining":      my_left,
            "opponent_balls_remaining": opp_left,
            "balls_on_table": {
                "solid":  self._stable.get("solid",  0),
                "stripe": self._stable.get("stripe", 0),
                "8ball":  self._stable.get("8ball",  0),
                "cue":    self._stable.get("cue",    0),
            },
            "8ball_on_table": self._stable.get("8ball", 0) >= 1,
            "cue_on_table":   self._stable.get("cue",   0) >= 1,
            "fouls": [e for e in self._events
                      if e["type"] in ("scratch", "foul_8ball")],
            "recent_events":  self._events[-5:],
        }

    def _write(self, state: dict) -> None:
        try:
            with open(GAME_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
