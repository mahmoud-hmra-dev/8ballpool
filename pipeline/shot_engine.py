"""
Shot physics engine.

Replaces the old physics.py + scattered shot logic in detector.py.

All geometry helpers are pure functions (no state, easy to test).
The two high-level selectors (shot_from_ghost, best_physics_shot) are
also pure — they take all inputs explicitly and return a Shot or None.

Score formula:  score = cos(cut_angle) / (miss_px + 1)
  cos(0°)  = 1.0  — straight-in shot, easiest
  cos(45°) = 0.71 — moderate cut
  cos(75°) = 0.26 — hard cut
  cos(90°) = 0.0  — physically impossible
  miss_px         — how far the target-ball ray misses the pocket centre;
                    lower = more accurate; shots above MAX_MISS_PX are rejected
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from config import BOUNCE_STEPS, GHOST_MIN_CONF, MAX_MISS_PX, RAY_LEN
from models import Shot, TableBounds, Vec2


# ── Vector math ───────────────────────────────────────────────────────────────
# Small inline helpers. Named with a leading underscore to signal they are
# internal utilities, not part of the public API.

def _v(a: Vec2, b: Vec2) -> Vec2:           return (b[0] - a[0], b[1] - a[1])
def _len(v: Vec2) -> float:                 return math.hypot(v[0], v[1])
def _dot(a: Vec2, b: Vec2) -> float:        return a[0] * b[0] + a[1] * b[1]
def _cross(a: Vec2, b: Vec2) -> float:      return a[0] * b[1] - a[1] * b[0]   # z of 3-D cross
def _add(p: Vec2, d: Vec2, t: float = 1.0) -> Vec2: return (p[0] + d[0]*t, p[1] + d[1]*t)
def _dist(a: Vec2, b: Vec2) -> float:       return _len(_v(a, b))

def _norm(v: Vec2) -> Vec2:
    m = _len(v)
    return (v[0] / m, v[1] / m) if m > 1e-9 else (0.0, 0.0)

def _line_dist(p: Vec2, a: Vec2, b: Vec2) -> float:
    """Perpendicular distance from point p to the infinite line through a→b."""
    d = _v(a, b)
    m = _dot(d, d)
    if m < 1e-12:
        return _dist(p, a)
    t = _dot(_v(a, p), d) / m
    return _dist(p, _add(a, d, t))


# ── Pure geometry ─────────────────────────────────────────────────────────────

def ghost_ball_pos(cue: Vec2, target: Vec2, ball_r: int) -> Vec2:
    """
    Ghost Ball Method: where the cue ball centre must be at the moment of
    impact with the target ball. One ball-diameter behind the target along
    the cue→target line.
    """
    d = _norm(_v(cue, target))
    return _add(target, d, -2 * ball_r)


def cut_angle_deg(cue: Vec2, target: Vec2, pocket: Vec2) -> float:
    """
    Angle between the aim line (cue→target) and the potting line (target→pocket).
    0° = straight-in (easy).  90° = impossible.
    """
    aim    = _norm(_v(cue, target))
    to_pkt = _norm(_v(target, pocket))
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(aim, to_pkt)))))


def cue_deflection(aim_dir: Vec2, target_dir: Vec2) -> Vec2:
    """
    90° Rule: after impact the cue ball deflects approximately 90° from
    the target ball's travel direction. The cross product sign tells us
    which perpendicular (left or right of target_dir) the cue ball takes.
    """
    cross = _cross(aim_dir, target_dir)
    if cross >= 0:
        return (-target_dir[1],  target_dir[0])   # 90° CCW
    return   ( target_dir[1], -target_dir[0])     # 90° CW


def trace_bounces(
    start:     Vec2,
    direction: Vec2,
    table:     TableBounds,
    R:         int = 0,
) -> list[Vec2]:
    """
    Ball-radius-aware wall-bounce path tracing.
    The ball centre bounces at ±R from each wall, not at the wall itself.
    Returns waypoints: [start, bounce1, bounce2, …].
    """
    # Effective wall positions (ball centre must stay R away from each wall)
    wx1, wy1 = table.x1 + R, table.y1 + R
    wx2, wy2 = table.x2 - R, table.y2 - R

    pts      = [start]
    x, y     = float(start[0]), float(start[1])
    dx, dy   = map(float, direction)

    for _ in range(BOUNCE_STEPS):
        cands = []
        if abs(dx) > 1e-9:
            cands.append(((wx2 - x) / dx if dx > 0 else (wx1 - x) / dx, "x"))
        if abs(dy) > 1e-9:
            cands.append(((wy2 - y) / dy if dy > 0 else (wy1 - y) / dy, "y"))
        if not cands:
            break
        t_min, axis = min(cands, key=lambda c: c[0] if c[0] > 1e-3 else 1e9)
        if t_min <= 1e-3:
            break
        x += dx * t_min; y += dy * t_min
        pts.append((x, y))
        if axis == "x": dx = -dx
        else:           dy = -dy

    return pts


# ── Shot builder ──────────────────────────────────────────────────────────────

def build_shot(
    cue_pos:    Vec2,
    ghost_pos:  Vec2,
    target_pos: Vec2,
    pockets:    list[Vec2],
    table:      TableBounds,
    R:          int,
    source:     str,
) -> Optional[Shot]:
    """
    Build a Shot from fully-known geometry.
    Returns None if no pocket is reachable within MAX_MISS_PX.

    This is the single place where shot geometry is assembled.
    Both shot_from_ghost and best_physics_shot funnel through here.
    """
    aim_dir    = _norm(_v(cue_pos, ghost_pos))
    target_dir = _norm(_v(ghost_pos, target_pos))
    cue_def    = cue_deflection(aim_dir, target_dir)

    # Score every pocket; keep the one whose line the target ball travels toward
    best_pocket, best_miss = None, float("inf")
    for pk in pockets:
        # Skip pockets that are behind the target ball's direction of travel
        if _dot(_v(target_pos, pk), target_dir) < 0:
            continue
        end  = _add(target_pos, target_dir, RAY_LEN)
        miss = _line_dist(pk, target_pos, end)
        if miss < best_miss:
            best_miss   = miss
            best_pocket = pk

    if best_pocket is None or best_miss > MAX_MISS_PX:
        return None

    cue_path = trace_bounces(ghost_pos, cue_def, table, R)
    angle    = cut_angle_deg(cue_pos, target_pos, best_pocket)
    score    = max(0.0, math.cos(math.radians(angle))) / (best_miss + 1.0)

    return Shot(
        cue_pos     = cue_pos,
        ghost_pos   = ghost_pos,
        target_pos  = target_pos,
        pocket      = best_pocket,
        target_dir  = target_dir,
        cue_def     = cue_def,
        cue_path    = cue_path,
        ball_radius = R,
        miss_px     = best_miss,
        cut_angle   = angle,
        score       = score,
        source      = source,
    )


# ── High-level shot selectors ─────────────────────────────────────────────────

def shot_from_ghost(
    cue_pos:    Vec2,
    ghost_pos:  Vec2,
    aim_conf:   float,
    balls:      list[dict],
    pockets:    list[Vec2],
    table:      TableBounds,
    R:          int,
) -> Optional[Shot]:
    """
    PATH A — player is actively aiming.

    The YOLO ghost ball gives us the exact ghost position. We infer the
    target ball as the non-cue ball closest to the predicted contact point
    (ghost + aim_direction × 2R).
    """
    if aim_conf < GHOST_MIN_CONF:
        return None

    aim_dir   = _norm(_v(cue_pos, ghost_pos))
    pred_pos  = _add(ghost_pos, aim_dir, 2 * R)   # predicted contact point

    candidates = [b for b in balls if b["type"] != "cue"]
    if not candidates:
        return None

    target = min(candidates, key=lambda b: _dist(b["pos"], pred_pos))
    if _dist(target["pos"], pred_pos) > R * 5:
        return None   # no ball near the predicted contact — ghost is noise

    return build_shot(cue_pos, ghost_pos, target["pos"], pockets, table, R, source="ai")


def best_physics_shot(
    cue_pos:  Vec2,
    balls:    list[dict],
    my_type:  Optional[str],
    pockets:  list[Vec2],
    table:    TableBounds,
    R:        int,
) -> Optional[Shot]:
    """
    PATH B — player is not aiming.

    Evaluate every (target ball, pocket) pair with the Ghost Ball Method
    and return the highest-score shot. Prefers the player's own ball type
    when it is known (set via the Solid/Stripe buttons in the overlay).
    """
    if my_type:
        candidates = [b for b in balls if b.get("subtype") == my_type]
    else:
        candidates = [b for b in balls if b["type"] != "cue"]

    # Graceful fallback: if ownership is unknown or no own balls are visible
    if not candidates:
        candidates = [b for b in balls if b["type"] != "cue"]

    best, best_score = None, -1.0
    for b in candidates:
        ghost = ghost_ball_pos(cue_pos, b["pos"], R)
        shot  = build_shot(cue_pos, ghost, b["pos"], pockets, table, R, source="calc")
        if shot and shot.score > best_score:
            best_score = shot.score
            best       = shot

    return best
