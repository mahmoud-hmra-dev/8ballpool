"""
Ball classifier.

Stateless pure functions — every function takes a patch (NumPy BGR array)
and returns a result. No side effects, easy to unit-test.

Ball types
  "cue"    — the white cue ball
  "8ball"  — the black 8 ball
  "solid"  — colored ball without a white stripe
  "stripe" — colored ball with a white stripe around the middle
  "ball"   — detected ball but color/type is ambiguous

Colors (returned as the 'type' string for colored balls)
  yellow, blue, red, purple, orange, green, maroon
"""
from __future__ import annotations

import cv2
import numpy as np

from config import (
    CUE_WHITENESS_THRESH, CUE_WHITE_FRAC,
    EIGHT_BALL_DARK_FRAC, EIGHT_BALL_V_THRESH,
    STRIPE_MID_Y0, STRIPE_MID_Y1, STRIPE_MID_WHITE_MIN, STRIPE_TOTAL_WHITE_MAX,
)

# HSV bounds for each color: ([H_lo, S_lo, V_lo], [H_hi, S_hi, V_hi])
_COLOR_RANGES: dict[str, tuple[list[int], list[int]]] = {
    "yellow": ([15, 100, 100], [35,  255, 255]),
    "blue":   ([90, 100,  70], [130, 255, 255]),
    "red":    ([0,  110,  70], [12,  255, 255]),
    "red2":   ([163, 110, 65], [180, 255, 255]),   # red wraps around hue=180
    "purple": ([120,  55, 45], [165, 255, 255]),
    "orange": ([10,  130, 95], [22,  255, 255]),
    "green":  ([35,   85, 45], [87,  255, 255]),
    "maroon": ([6,    40, 28], [18,  150, 120]),
}


def whiteness_score(patch: np.ndarray, radius: int) -> float:
    """
    How white is the ball's centre region?
    Returns 0.0–1.0. Values > 0.22 reliably indicate the cue ball.

    Measures the fraction of pixels in the central circle that are
    high-value (bright) AND low-saturation (white-ish), weighted by
    the mean brightness. This handles lighting variation gracefully.
    """
    if patch is None or patch.size == 0:
        return 0.0
    hsv    = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    r      = max(3, int(radius * 0.85))
    h, w   = hsv.shape[:2]
    cy, cx = h // 2, w // 2
    center = hsv[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    if center.size == 0:
        return 0.0
    v, s = center[:, :, 2], center[:, :, 1]
    n    = center.shape[0] * center.shape[1]
    white_frac = float(np.sum((v > 170) & (s < 70))) / max(n, 1)
    brightness = float(np.mean(v)) / 255.0
    return white_frac * brightness


def is_stripe(patch: np.ndarray) -> bool:
    """
    Stripe balls have a white equatorial band; solid balls don't.
    Strategy: white concentration in the centre band >> top/bottom bands.
    """
    if patch is None or patch.size == 0:
        return False
    hsv    = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hh, ww = hsv.shape[:2]
    if hh < 8:
        return False

    # Three horizontal bands
    y_top = int(hh * 0.25)
    y_bot = int(hh * 0.75)
    top  = hsv[:y_top, :]
    mid  = hsv[y_top:y_bot, :]
    bot  = hsv[y_bot:, :]

    def white_frac(region):
        if region.size == 0:
            return 0.0
        v = region[:, :, 2].flatten()
        s = region[:, :, 1].flatten()
        return float(np.sum((v > 150) & (s < 100))) / max(len(v), 1)

    wf_mid = white_frac(mid)
    wf_top = white_frac(top)
    wf_bot = white_frac(bot)
    wf_all = white_frac(hsv)

    # Stripe: plenty of white in the centre, AND centre has clearly more
    # white than the top/bottom, AND not so much total white it's the cue
    mid_dominant = wf_mid > max(wf_top, wf_bot) * 1.4 + 0.05
    return wf_mid > STRIPE_MID_WHITE_MIN and mid_dominant and wf_all < STRIPE_TOTAL_WHITE_MAX


def classify_color(patch: np.ndarray) -> str:
    """
    Return the dominant color name of the ball, or 'ball' if ambiguous.
    Also returns 'cue' or '8ball' for special cases.
    """
    if patch is None or patch.size == 0:
        return "ball"
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h   = hsv[:, :, 0].flatten().astype(float)
    s   = hsv[:, :, 1].flatten().astype(float)
    v   = hsv[:, :, 2].flatten().astype(float)
    n   = len(h)

    # Special cases first (before color matching)
    if np.sum((v > 170) & (s < 70)) / n > CUE_WHITE_FRAC:
        return "cue"
    # 8-ball: mostly dark with no strong color (black body + white "8" circle)
    dark_and_gray = np.sum((v < EIGHT_BALL_V_THRESH) & (s < 90)) / n
    if dark_and_gray > EIGHT_BALL_DARK_FRAC and float(np.mean(s)) < 70:
        return "8ball"

    # Need meaningful saturation to identify a color
    sat_mask = (s > 80) & (v > 60)
    if sat_mask.sum() < max(4, int(n * 0.05)):
        return "ball"

    hm = float(np.median(h[sat_mask]))
    sm = float(np.median(s[sat_mask]))
    vm = float(np.median(v[sat_mask]))

    for name, (lo, hi) in _COLOR_RANGES.items():
        if lo[0] <= hm <= hi[0] and sm >= lo[1] and vm >= lo[2]:
            return name
    return "ball"


def classify_ball(patch: np.ndarray, radius: int) -> tuple[str, str]:
    """
    Full classification: returns (type, subtype).

    type    : color string, "cue", "8ball", or "ball" (unknown)
    subtype : "solid" or "stripe" for colored balls; same as type otherwise
    """
    color = classify_color(patch)
    if color in ("cue", "8ball", "ball"):
        return color, color
    subtype = "stripe" if is_stripe(patch) else "solid"
    return color, subtype
