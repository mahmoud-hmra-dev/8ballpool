"""
Pool shot physics — Ghost Ball method.

Score formula:
    score = cos(cut_angle) / (miss_px + 1)

  • cos(0°)  = 1.0  → straight-in shot is easiest
  • cos(45°) = 0.71 → moderate cut
  • cos(75°) = 0.26 → hard cut
  • cos(90°) = 0.0  → impossible (90° cut)
  • miss_px  = perpendicular distance from pocket centre to the
               projected target-ball ray (lower = more accurate)

Only shots with miss_px < MAX_MISS are considered pocketable.
"""
import math
from typing import Optional

Vec2 = tuple[float, float]

MAX_MISS   = 55    # px – reject shots where ball misses pocket by more than this
SEARCH_LEN = 1500  # px – how far to extend the target-ball ray when measuring miss


# ── vector helpers ─────────────────────────────────────────────────────────────

def dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def norm(v: Vec2) -> Vec2:
    m = math.hypot(v[0], v[1])
    return (v[0] / m, v[1] / m) if m > 1e-9 else (0.0, 0.0)


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def point_to_line_dist(p: Vec2, a: Vec2, b: Vec2) -> float:
    """Perpendicular distance from point p to the INFINITE line through a→b."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    if ax == 0 and ay == 0:
        return dist(p, a)
    t       = ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / (ax * ax + ay * ay)
    closest = (a[0] + t * ax, a[1] + t * ay)
    return dist(p, closest)


# ── core geometry ──────────────────────────────────────────────────────────────

def ghost_ball_pos(cue: Vec2, target: Vec2, ball_r: int) -> Vec2:
    """
    Position where the CUE BALL centre must be at the moment of impact.
    One ball-diameter behind the target in the cue→target direction.
    """
    nx, ny = norm((target[0] - cue[0], target[1] - cue[1]))
    return (target[0] - nx * 2 * ball_r,
            target[1] - ny * 2 * ball_r)


def cut_angle_deg(cue: Vec2, target: Vec2, pocket: Vec2) -> float:
    """
    Angle (°) between the aim line (cue→target) and the potting line
    (target→pocket).  0° = straight-in.  90° = impossible.
    """
    aim_dir    = norm((target[0] - cue[0],    target[1] - cue[1]))
    pocket_dir = norm((pocket[0] - target[0], pocket[1] - target[1]))
    cos_a      = max(-1.0, min(1.0, dot(aim_dir, pocket_dir)))
    return math.degrees(math.acos(cos_a))


def cue_deflection(cue: Vec2, target: Vec2) -> Vec2:
    """
    Cue ball deflects ~90° from the target ball's path (90° rule).
    Returns the unit direction vector for the cue ball after impact.
    """
    tx, ty = norm((target[0] - cue[0], target[1] - cue[1]))
    return (-ty, tx)   # 90° CCW rotation


def trace_path(start: Vec2, direction: Vec2,
               table: tuple, steps: int = 4) -> list[Vec2]:
    """
    Bounce a ray off the table walls.
    Returns waypoints [start, bounce1, bounce2, …].
    table = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = table
    pts = [start]
    x, y   = float(start[0]), float(start[1])
    dx, dy = direction

    for _ in range(steps):
        candidates = []
        if abs(dx) > 1e-9:
            t = (x2 - x) / dx if dx > 0 else (x1 - x) / dx
            candidates.append((t, "x"))
        if abs(dy) > 1e-9:
            t = (y2 - y) / dy if dy > 0 else (y1 - y) / dy
            candidates.append((t, "y"))
        if not candidates:
            break
        t_min, axis = min(candidates, key=lambda c: c[0] if c[0] > 1e-3 else 1e9)
        if t_min <= 1e-3:
            break
        x += dx * t_min
        y += dy * t_min
        pts.append((x, y))
        if axis == "x":
            dx = -dx
        else:
            dy = -dy

    return pts


# ── shot selection ─────────────────────────────────────────────────────────────

def best_shot(
    cue_pos:      Vec2,
    target_balls: list[dict],
    pockets:      list[Vec2],
    ball_r:       int,
    table_bounds: tuple,
) -> Optional[dict]:
    """
    Evaluate every (target ball, pocket) pair and return the highest-score shot.

    Result dict keys
    ----------------
    cue_pos, target_pos, ghost_pos, pocket
    target_dir, cue_deflect, cue_path
    ball_radius, miss_px, cut_angle, score
    """
    best       = None
    best_score = -1.0

    for ball in target_balls:
        tpos  = ball["pos"]
        tdir  = norm((tpos[0] - cue_pos[0], tpos[1] - cue_pos[1]))
        ghost = ghost_ball_pos(cue_pos, tpos, ball_r)
        cue_d = cue_deflection(cue_pos, tpos)

        for pocket in pockets:
            # Only pockets that are ahead of the target ball
            t_proj = ((pocket[0] - tpos[0]) * tdir[0] +
                      (pocket[1] - tpos[1]) * tdir[1])
            if t_proj < 0:
                continue

            # Extend target-ball ray and measure miss distance
            end = (tpos[0] + tdir[0] * SEARCH_LEN,
                   tpos[1] + tdir[1] * SEARCH_LEN)
            miss = point_to_line_dist(pocket, tpos, end)

            if miss > MAX_MISS:
                continue

            # Cut angle (0° = straight, 90° = impossible)
            angle = cut_angle_deg(cue_pos, tpos, pocket)
            if angle >= 88:
                continue

            # Score: straight easy shots rank highest
            angle_factor = max(0.0, math.cos(math.radians(angle)))
            score        = angle_factor / (miss + 1.0)

            if score > best_score:
                best_score = score
                cue_path   = trace_path(ghost, cue_d, table_bounds)

                best = {
                    "cue_pos":     cue_pos,
                    "target_pos":  tpos,
                    "ghost_pos":   ghost,
                    "pocket":      pocket,
                    "target_dir":  tdir,
                    "cue_deflect": cue_d,
                    "cue_path":    cue_path,
                    "ball_radius": ball_r,
                    "miss_px":     miss,
                    "cut_angle":   angle,
                    "score":       score,
                }

    return best
