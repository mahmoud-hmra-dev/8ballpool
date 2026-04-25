"""
Central configuration.

Every tunable constant lives here. Change one value — it propagates
everywhere. No more hunting through 5 files for a threshold.
"""

# ── Pipeline ──────────────────────────────────────────────────────────────────
TARGET_FPS         = 60
TABLE_RECALC       = 90    # frames between full table re-detections
BROWSER_TOP_OFFSET = 90    # px to skip (Chrome title + tabs + address bar)

# ── YOLO ──────────────────────────────────────────────────────────────────────
BALL_CONF   = 0.30   # minimum confidence to accept a ball detection
COLL_CONF   = 0.35   # minimum confidence to accept a collision (ghost ball) detection
YOLO_IMGSZ  = 640    # YOLO input resolution; reduce to 416 for more speed (less accuracy)

# ── Ball tracking (Exponential Weighted Average) ──────────────────────────────
# TRACK_ALPHA: how quickly a track follows a new detection.
#   0.0 = frozen,  1.0 = raw (jittery).  0.35 is a good balance.
TRACK_ALPHA    = 0.35
TRACK_MAX_DIST = 45    # px — max distance to match detection to an existing track
TRACK_EXPIRE   = 8     # frames before an unseen track is dropped (ball pocketed)

# ── Ghost ball stability ───────────────────────────────────────────────────────
GHOST_BUF_LEN  = 6     # rolling median window (frames)
GHOST_MIN_CONF = 0.50  # minimum YOLO confidence to add to the stable buffer
GHOST_STALE    = 15    # frames without a detection before the buffer is cleared

# ── Physics / shot selection ──────────────────────────────────────────────────
MAX_MISS_PX  = 65     # reject shots where target-ball ray misses pocket by more
RAY_LEN      = 1800   # px — how far to extend the target-ball ray when scoring
BOUNCE_STEPS = 5      # max wall bounces to trace for the cue-ball path

# ── Table detection ───────────────────────────────────────────────────────────
SKIP_TOP_FRAC        = 0.27   # Chrome: fraction of frame height to skip (UI panel at top)
SKIP_TOP_FRAC_SCRCPY = 0.22   # scrcpy: skip titlebar + game UI header
SCRCPY_EXPAND_PX     = 39     # px to expand table bounds outward after edge refinement

# ── Ball classification ────────────────────────────────────────────────────────
CUE_WHITENESS_THRESH   = 0.22   # whiteness_score() threshold to identify the cue ball
CUE_WHITE_FRAC         = 0.35   # fraction of high-V/low-S pixels → classify as cue
EIGHT_BALL_DARK_FRAC   = 0.55   # fraction of dark pixels → classify as 8-ball
STRIPE_MID_Y0          = 0.33   # stripe middle-band top boundary (fraction of patch height)
STRIPE_MID_Y1          = 0.67   # stripe middle-band bottom boundary
STRIPE_MID_WHITE_MIN   = 0.22   # white fraction in middle band must exceed this → stripe
STRIPE_TOTAL_WHITE_MAX = 0.42   # overall white fraction must stay below this → stripe (not cue)

# ── Shot difficulty thresholds (used by renderer) ─────────────────────────────
DIFF_EASY_MISS_PX    = 12   # miss ≤ this AND angle ≤ DIFF_EASY_ANGLE_DEG  → EASY
DIFF_EASY_ANGLE_DEG  = 30
DIFF_MED_MISS_PX     = 35   # miss ≤ this AND angle ≤ DIFF_MED_ANGLE_DEG   → MEDIUM
DIFF_MED_ANGLE_DEG   = 55
