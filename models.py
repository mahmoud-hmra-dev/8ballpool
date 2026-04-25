"""
Data models for the shot assistant pipeline.

Using dataclasses makes each stage's contract explicit and self-documenting.
Every field has a name and type — no more mystery dict keys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

Vec2 = tuple[float, float]   # (x, y) in screen pixels


# ── Ball ──────────────────────────────────────────────────────────────────────

@dataclass
class Ball:
    pos:    Vec2
    radius: int
    kind:   Literal["cue", "eight", "solid", "stripe", "ball"]
    color:  str = "unknown"


# ── Table ─────────────────────────────────────────────────────────────────────

@dataclass
class TableBounds:
    x1: int
    y1: int
    x2: int
    y2: int
    ball_radius: int = 13

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"TableBounds degenerate: ({self.x1},{self.y1})→({self.x2},{self.y2})"
            )
        self.ball_radius = max(5, min(40, self.ball_radius))

    # ── convenience ───────────────────────────────────────────────────────────

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def pockets(self) -> list[Vec2]:
        """
        Six standard pocket positions: top-left, top-mid, top-right,
        bottom-left, bottom-mid, bottom-right.
        """
        ci = max(6, self.ball_radius)   # corner inset
        mx = (self.x1 + self.x2) // 2
        return [
            (self.x1 + ci, self.y1 + ci), (mx, self.y1), (self.x2 - ci, self.y1 + ci),
            (self.x1 + ci, self.y2 - ci), (mx, self.y2), (self.x2 - ci, self.y2 - ci),
        ]


# ── Shot ──────────────────────────────────────────────────────────────────────

@dataclass
class Shot:
    cue_pos:     Vec2
    ghost_pos:   Vec2           # where cue ball must be at impact
    target_pos:  Vec2
    pocket:      Vec2
    target_dir:  Vec2           # unit vector: target ball travels this way
    cue_def:     Vec2           # unit vector: cue ball deflects this way (90° rule)
    cue_path:    list[Vec2]     # waypoints: ghost → wall bounces
    ball_radius: int
    miss_px:     float          # ⊥ distance from pocket to target-ball ray
    cut_angle:   float          # degrees (0° = straight-in, 90° = impossible)
    score:       float          # cos(cut_angle) / (miss_px + 1)
    source:      Literal["ai", "calc"]  # "ai" = YOLO ghost,  "calc" = physics

    def translate(self, dx: float, dy: float) -> Shot:
        """
        Return a new Shot with every position shifted by (-dx, -dy).
        Used to convert frame coordinates → overlay canvas coordinates.
        """
        def _t(p: Vec2) -> Vec2:
            return (p[0] - dx, p[1] - dy)

        return Shot(
            cue_pos     = _t(self.cue_pos),
            ghost_pos   = _t(self.ghost_pos),
            target_pos  = _t(self.target_pos),
            pocket      = _t(self.pocket),
            target_dir  = self.target_dir,   # direction vectors don't translate
            cue_def     = self.cue_def,
            cue_path    = [_t(p) for p in self.cue_path],
            ball_radius = self.ball_radius,
            miss_px     = self.miss_px,
            cut_angle   = self.cut_angle,
            score       = self.score,
            source      = self.source,
        )
