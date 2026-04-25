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
BALL_CONF  = 0.30   # minimum confidence to accept a ball detection
COLL_CONF  = 0.35   # minimum confidence to accept a collision (ghost ball) detection

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
SKIP_TOP_FRAC = 0.27  # fraction of frame height to skip (score/UI panel at top)
