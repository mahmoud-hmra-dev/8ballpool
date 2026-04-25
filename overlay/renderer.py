"""
Shot overlay renderer.

Pure drawing — no state except the canvas reference.
Receives a Shot dataclass (or None) and renders 8 visual layers onto
a Tkinter canvas. Nothing here knows about threads, queues, or windows.

Layers (bottom → top)
  1  Aim line       cue → ghost (solid white) + extended (dashed white)
  2  Cue path       ghost → wall bounces (blue→dim fading segments)
  3  Target path    target → pocket (difficulty-colored arrow)
  4  Pocket ring    pulsing animated circle
  5  Ghost ball     dashed circle + crosshair
  6  Cue highlight  yellow ring around the cue ball
  7  Target ring    difficulty-colored ring around the target ball
  8  HUD            [AI/CALC]  difficulty  miss  cut  score
"""
from __future__ import annotations

import math
import tkinter as tk
from typing import Optional

from models import Shot, Vec2

# Cue-ball bounce path: bright blue fading to dark as the ball travels further
_BOUNCE_COLS = ["#44AAFF", "#3399EE", "#2277CC", "#1155AA", "#003388"]


def _difficulty_color(miss: float, angle: float) -> str:
    """Shot difficulty encoded as a color — used on every difficulty-colored element."""
    if miss < 12 and angle < 30: return "#00FF88"   # easy   — green
    if miss < 35 and angle < 55: return "#FFD700"   # medium — gold
    return "#FF4444"                                  # hard   — red


def _difficulty_label(miss: float, angle: float) -> str:
    if miss < 12 and angle < 30: return "EASY"
    if miss < 35 and angle < 55: return "MEDIUM"
    return "HARD"


class Renderer:
    """Draws one shot per frame onto a Tkinter canvas."""

    def __init__(self, canvas: tk.Canvas):
        self._c = canvas

    def draw(self, shot: Optional[Shot], tick: int) -> None:
        """
        Clear the canvas and render the shot (or nothing if shot is None).
        tick drives the pocket-ring pulse animation (0–39 cycling at 60 Hz).
        """
        c = self._c
        c.delete("all")
        if shot is None:
            return

        R    = max(8, shot.ball_radius)
        col  = _difficulty_color(shot.miss_px, shot.cut_angle)
        diff = _difficulty_label(shot.miss_px, shot.cut_angle)

        self._aim_line(shot.cue_pos, shot.ghost_pos, R)
        self._cue_path(shot.cue_path)
        self._target_path(shot.target_pos, shot.pocket, col)
        self._pocket_ring(shot.pocket, col, tick)
        self._ghost_ball(shot.ghost_pos, R)
        self._cue_highlight(shot.cue_pos, R)
        self._target_highlight(shot.target_pos, R, col)
        self._hud(shot, col, diff)

    # ── Layer 1: aim line ─────────────────────────────────────────────────────

    def _aim_line(self, cue: Vec2, ghost: Vec2, R: int) -> None:
        if not (cue and ghost):
            return
        dx, dy = ghost[0] - cue[0], ghost[1] - cue[1]
        m      = max(1.0, math.hypot(dx, dy))
        nx, ny = dx / m, dy / m
        sx, sy = int(cue[0] + nx * (R + 3)), int(cue[1] + ny * (R + 3))

        # Solid segment: cue ball → ghost position (where cue ball travels)
        self._c.create_line(sx, sy, int(ghost[0]), int(ghost[1]),
                            fill="white", width=2)
        # Dashed extension: ghost → +900 px (aim direction beyond contact)
        ex = int(ghost[0] + nx * 900); ey = int(ghost[1] + ny * 900)
        self._c.create_line(int(ghost[0]), int(ghost[1]), ex, ey,
                            fill="white", width=1, dash=(10, 7))

    # ── Layer 2: cue-ball bounce path ─────────────────────────────────────────

    def _cue_path(self, cpath: list[Vec2]) -> None:
        if not cpath or len(cpath) < 2:
            return
        for i in range(len(cpath) - 1):
            p0, p1  = cpath[i], cpath[i + 1]
            seg_col = _BOUNCE_COLS[min(i, len(_BOUNCE_COLS) - 1)]
            self._c.create_line(
                int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]),
                fill=seg_col, width=max(1, 3 - i), dash=(8, 4) if i == 0 else (5, 5))
        # Arrow on the last segment to show travel direction
        p0, p1 = cpath[-2], cpath[-1]
        self._c.create_line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]),
                            fill=_BOUNCE_COLS[0], width=2,
                            arrow=tk.LAST, arrowshape=(10, 12, 4))

    # ── Layer 3: target ball → pocket ─────────────────────────────────────────

    def _target_path(self, target: Vec2, pocket: Vec2, col: str) -> None:
        if not (target and pocket):
            return
        self._c.create_line(int(target[0]), int(target[1]),
                            int(pocket[0]),  int(pocket[1]),
                            fill=col, width=3, dash=(10, 5),
                            arrow=tk.LAST, arrowshape=(15, 18, 6))

    # ── Layer 4: pocket ring (animated pulse) ─────────────────────────────────

    def _pocket_ring(self, pocket: Vec2, col: str, tick: int) -> None:
        if not pocket:
            return
        px, py = int(pocket[0]), int(pocket[1])
        pr     = 16 + int(5 * math.sin(tick * math.pi / 20))   # pulsing radius
        self._c.create_oval(px-pr-3, py-pr-3, px+pr+3, py+pr+3, outline=col, width=1)
        self._c.create_oval(px-pr,   py-pr,   px+pr,   py+pr,   outline=col, width=3)
        self._c.create_oval(px-4,    py-4,    px+4,    py+4,    fill=col, outline="")

    # ── Layer 5: ghost ball (dashed circle + crosshair) ───────────────────────

    def _ghost_ball(self, ghost: Vec2, R: int) -> None:
        if not ghost:
            return
        gx, gy = int(ghost[0]), int(ghost[1])
        self._c.create_oval(gx-R, gy-R, gx+R, gy+R,
                            outline="white", width=2, dash=(4, 3))
        self._c.create_line(gx-R+2, gy, gx+R-2, gy, fill="white", width=1)
        self._c.create_line(gx, gy-R+2, gx, gy+R-2, fill="white", width=1)

    # ── Layer 6: cue ball highlight ───────────────────────────────────────────

    def _cue_highlight(self, cue: Vec2, R: int) -> None:
        if not cue:
            return
        cx, cy = int(cue[0]), int(cue[1])
        self._c.create_oval(cx-R-5, cy-R-5, cx+R+5, cy+R+5,
                            outline="#FFFF00", width=3)

    # ── Layer 7: target ball highlight ────────────────────────────────────────

    def _target_highlight(self, target: Vec2, R: int, col: str) -> None:
        if not target:
            return
        tx, ty = int(target[0]), int(target[1])
        self._c.create_oval(tx-R-5, ty-R-5, tx+R+5, ty+R+5,
                            outline=col, width=3)

    # ── Layer 8: HUD ──────────────────────────────────────────────────────────

    def _hud(self, shot: Shot, col: str, diff: str) -> None:
        src = "AI" if shot.source == "ai" else "CALC"
        hud = (
            f"  [{src}]  {diff}   "
            f"miss={shot.miss_px:.0f}px   cut={shot.cut_angle:.0f}°   "
            f"score={shot.score:.3f}"
        )
        self._c.create_rectangle(5, 5, 400, 29, fill="#111111", outline=col, width=1)
        self._c.create_text(12, 17, anchor="w", text=hud,
                            fill=col, font=("Consolas", 11, "bold"))
