"""
Full AI-based shot assistant  —  best.pt  (ball / collision)
=============================================================

YOLO classes
  0  ball       → position of every ball on the table
  1  collision  → ghost-ball position (where the cue-ball centre must be
                  at the moment of impact with the target ball)

What we derive from those two detections
  ┌─ cue ball     whitest class-0 ball
  ├─ ghost pos    highest-confidence class-1 box
  ├─ aim dir      normalize(ghost − cue)
  ├─ target ball  class-0 ball closest to (ghost + aim_dir × 2r)
  ├─ target dir   normalize(target − ghost)   [direction target will travel]
  ├─ best pocket  pocket that the target-ball ray passes closest to
  ├─ cue deflect  90°-rule: perpendicular to target_dir (correct side from cut)
  └─ cue path     trace from ghost with wall bounces (ball-radius-aware)

When no collision is detected (player not aiming) we fall back to the
best-shot physics calculator so the overlay always shows something useful.
"""

import math
import os
from collections import deque
import cv2
import numpy as np

try:
    import torch
    _DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _HALF   = torch.cuda.is_available()
except ImportError:
    _DEVICE = 'cpu'
    _HALF   = False

# ── tracking / temporal stability ────────────────────────────────────────────
_TRACK_ALPHA    = 0.35   # EWA weight for new position  (0=never update, 1=raw)
_TRACK_MAX_DIST = 45     # px — max dist to match detection → existing track
_TRACK_EXPIRE   = 8      # frames before unseen track is deleted
_GHOST_BUF_LEN  = 6      # ghost ball history length (frames)
_GHOST_MIN_CONF = 0.50   # min collision conf to add to stable-ghost buffer
_GHOST_STALE    = 15     # frames without ghost before buffer is cleared

_SKIP_TOP_FRAC = 0.27
_MODEL_DIR     = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH   = os.path.join(_MODEL_DIR, "best.engine")
_PT_PATH       = os.path.join(_MODEL_DIR, "best.pt")
# Auto-use TensorRT engine if it exists (3-5× faster than fp16 YOLO)
_YOLO_PATH     = _ENGINE_PATH if os.path.exists(_ENGINE_PATH) else _PT_PATH
_BALL_CONF     = 0.30
_COLL_CONF     = 0.35
_MAX_MISS      = 65      # px – max distance from pocket to target ray
_RAY_LEN       = 1800    # px – how far to extend target-ball ray
_BOUNCE_STEPS  = 5       # number of cue-ball wall bounces to trace


# ── micro vector library ──────────────────────────────────────────────────────

def _v(a, b):        return (b[0]-a[0], b[1]-a[1])
def _len(v):         return math.hypot(v[0], v[1])
def _norm(v):
    m = _len(v)
    return (v[0]/m, v[1]/m) if m > 1e-9 else (0.0, 0.0)
def _dot(a, b):      return a[0]*b[0] + a[1]*b[1]
def _cross2d(a, b):  return a[0]*b[1] - a[1]*b[0]   # z-component of 3-D cross
def _add(a, v, t=1): return (a[0]+v[0]*t, a[1]+v[1]*t)
def _dist(a, b):     return _len(_v(a, b))

def _line_dist(p, a, b):
    """⊥ distance from point p to infinite line a→b."""
    d = _v(a, b)
    m = _dot(d, d)
    if m < 1e-12:
        return _dist(p, a)
    t = _dot(_v(a, p), d) / m
    return _dist(p, _add(a, d, t))


class BallDetector:

    def __init__(self):
        self.table_bounds: tuple | None = None
        self.pockets:      list  | None = None
        self.ball_radius:  int   = 13
        self._my_type:     str   | None = None   # 'solid' | 'stripe' | None

        # ── temporal tracking ─────────────────────────────────────────────────
        self._tracks:      dict  = {}            # tid → track dict
        self._next_tid:    int   = 0
        self._frame_n:     int   = 0
        self._ghost_buf:   deque = deque(maxlen=_GHOST_BUF_LEN)
        self._ghost_frame: int   = -999          # last frame a ghost was recorded
        self.aim_conf:     float = 0.0           # current aiming confidence (0-1)

        if not os.path.exists(_ENGINE_PATH):
            print("[Detector] TRT engine not found — using best.pt\n"
                  "  For 3-5× speedup run once:\n"
                  f"  python -c \"from ultralytics import YOLO; "
                  f"YOLO(r'{_PT_PATH}').export(format='engine', half=True, device=0)\"")

        self._yolo = None
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(_YOLO_PATH)
            # Warm-up on the target device (compiles CUDA kernels on first call)
            self._yolo(np.zeros((64, 64, 3), np.uint8),
                       verbose=False, device=_DEVICE, half=_HALF)
            engine_tag = "TRT" if _YOLO_PATH.endswith(".engine") else "PT"
            print(f"[Detector] YOLO ready  [{engine_tag}]  device={_DEVICE}  "
                  f"half={_HALF}  {self._yolo.names}")
        except Exception as e:
            print(f"[Detector] YOLO failed: {e}")

    def set_my_type(self, t: str | None):
        """Set which ball subtype belongs to this player ('solid' | 'stripe' | None)."""
        self._my_type = t

    # ═══════════════════════════════════════════════════════════════════════════
    # BALL TRACKING  —  nearest-neighbour match + EWA smoothing
    # ═══════════════════════════════════════════════════════════════════════════

    def _match_tracks(self, detections: list) -> list:
        """
        Match new per-frame detections to existing tracks.
        Applies EWA to positions → no flickering.
        Returns list of active track dicts with smoothed 'pos'.
        """
        self._frame_n += 1
        used_tids = set()

        for det in detections:
            rx, ry = det["abs_xy"]
            # Find closest unmatched track
            best_tid, best_d = None, _TRACK_MAX_DIST
            for tid, tr in self._tracks.items():
                if tid in used_tids:
                    continue
                d = _dist(tr["pos"], (rx, ry))
                if d < best_d:
                    best_d = d; best_tid = tid

            a = _TRACK_ALPHA
            if best_tid is not None:
                tr = self._tracks[best_tid]
                tr["pos"]     = (tr["pos"][0]*(1-a) + rx*a,
                                 tr["pos"][1]*(1-a) + ry*a)
                tr["r"]       = int(tr["r"]*0.8 + det["r"]*0.2)
                tr["type"]    = det["type"]
                tr["subtype"] = det["subtype"]
                tr["ws"]      = det.get("ws", 0.0)
                tr["patch"]   = det["patch"]
                tr["last"]    = self._frame_n
                used_tids.add(best_tid)
            else:
                tid = self._next_tid; self._next_tid += 1
                self._tracks[tid] = dict(
                    pos=(float(rx), float(ry)),
                    r=det["r"], type=det["type"], subtype=det["subtype"],
                    ws=det.get("ws", 0.0), patch=det["patch"],
                    last=self._frame_n)
                used_tids.add(tid)

        # Expire stale tracks (ball was pocketed or mis-detected)
        stale = [tid for tid, tr in self._tracks.items()
                 if self._frame_n - tr["last"] > _TRACK_EXPIRE]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    # ═══════════════════════════════════════════════════════════════════════════
    # GHOST BALL STABILITY  —  median of recent high-confidence detections
    # ═══════════════════════════════════════════════════════════════════════════

    def _stable_ghost(self, raw_colls: list):
        """
        Return a temporally stable ghost ball pos (median of last N frames).
        Also updates self.aim_conf (0-1) so the overlay knows if player aims.
        """
        if raw_colls:
            best = max(raw_colls, key=lambda c: c["conf"])
            self.aim_conf = best["conf"]
            if best["conf"] >= _GHOST_MIN_CONF:
                self._ghost_buf.append(best["abs_xy"])
                self._ghost_frame = self._frame_n
        else:
            self.aim_conf = 0.0

        # Clear buffer when player stopped aiming for a while
        if self._frame_n - self._ghost_frame > _GHOST_STALE:
            self._ghost_buf.clear()

        if not self._ghost_buf:
            return None
        xs = [g[0] for g in self._ghost_buf]
        ys = [g[1] for g in self._ghost_buf]
        return (int(np.median(xs)), int(np.median(ys)))

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE DETECTION  (felt inner bounds — unchanged)
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_table(self, frame: np.ndarray) -> tuple | None:
        fh, fw = frame.shape[:2]
        skip   = int(fh * _SKIP_TOP_FRAC)
        search = frame[skip:, :]
        sc     = 0.5
        sm     = cv2.resize(search, None, fx=sc, fy=sc,
                            interpolation=cv2.INTER_AREA)
        hsv    = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)

        outer = (self._felt(hsv) or self._sat(hsv) or self._edges(sm))

        if outer is None:
            ox1=int(fw*.30); ox2=int(fw*.85)
            oy1=skip+int((fh-skip)*.05); oy2=int(fh*.90)
        else:
            sx,sy,sw,sh = outer
            ox1=int(sx/sc); oy1=int(sy/sc)+skip
            ox2=int((sx+sw)/sc); oy2=int((sy+sh)/sc)+skip

        ox1=max(0,ox1); oy1=max(0,oy1)
        ox2=min(fw,ox2); oy2=min(fh,oy2)

        r = self._refine(frame[oy1:oy2, ox1:ox2])
        if r:
            fx,fy,fw2,fh2 = r
            x1=ox1+fx; y1=oy1+fy; x2=x1+fw2; y2=y1+fh2
        else:
            tw=ox2-ox1; th=oy2-oy1
            x1=ox1+max(8,int(tw*.07)); y1=oy1+max(8,int(th*.10))
            x2=ox2-max(8,int(tw*.07)); y2=oy2-max(8,int(th*.10))

        inner = self._inner_edges(frame[y1:y2, x1:x2])
        if inner is not None:
            ix1, iy1, ix2, iy2 = inner
            x1 += ix1; y1 += iy1; x2 = x1 + (ix2 - ix1); y2 = y1 + (iy2 - iy1)

        x1=max(0,x1); y1=max(0,y1); x2=min(fw,x2); y2=min(fh,y2)
        tw=x2-x1; th=y2-y1
        self.ball_radius = max(7, min(30, max(int(tw/96), int(th/48))))
        inset = max(2, int(round(self.ball_radius * 0.15)))
        self.table_bounds = (x1+inset, y1+inset, x2-inset, y2-inset)
        return self.table_bounds

    def _felt(self, hsv):
        m  = cv2.inRange(hsv, np.array([78,50,65]),  np.array([118,210,225]))
        m |= cv2.inRange(hsv, np.array([55,45,55]),  np.array([80, 195,205]))
        k  = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        return self._brect(cv2.morphologyEx(
               cv2.morphologyEx(m,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k), 0.08)

    def _sat(self, hsv):
        m = cv2.inRange(hsv, np.array([0,40,35]), np.array([180,255,215]))
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        return self._brect(cv2.morphologyEx(
               cv2.morphologyEx(m,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k), 0.12)

    def _edges(self, sm):
        g = cv2.Canny(cv2.GaussianBlur(
            cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY),(5,5),1),25,80)
        g = cv2.dilate(g,cv2.getStructuringElement(
            cv2.MORPH_RECT,(3,3)),iterations=2)
        return self._brect(g, 0.10)

    def _refine(self, roi):
        if roi.size == 0: return None
        sc=0.5
        sm=cv2.resize(roi,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
        h =cv2.cvtColor(sm,cv2.COLOR_BGR2HSV)
        m  = cv2.inRange(h, np.array([78,50,65]),  np.array([118,210,225]))
        m |= cv2.inRange(h, np.array([55,45,55]),  np.array([80, 195,205]))
        k  = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        r  = self._brect(cv2.morphologyEx(
             cv2.morphologyEx(m,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k), 0.06)
        if r is None: return None
        x,y,w,hh=r
        return (int(x/sc),int(y/sc),int(w/sc),int(hh/sc))

    def _inner_edges(self, roi):
        if roi.size == 0:
            return None

        h, w = roi.shape[:2]
        if w < 80 or h < 60:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))

        band_y1 = int(h * 0.18)
        band_y2 = int(h * 0.82)
        band_x1 = int(w * 0.18)
        band_x2 = int(w * 0.82)
        if band_y2 <= band_y1 or band_x2 <= band_x1:
            return None

        vprof = gx[band_y1:band_y2].mean(axis=0).astype(np.float32)
        hprof = gy[:, band_x1:band_x2].mean(axis=1).astype(np.float32)
        vprof = cv2.GaussianBlur(vprof.reshape(1, -1), (1, 31), 0).ravel()
        hprof = cv2.GaussianBlur(hprof.reshape(-1, 1), (31, 1), 0).ravel()

        margin = 8
        xwin = max(24, int(w * 0.12))
        ywin = max(24, int(h * 0.12))
        if w <= 2 * margin or h <= 2 * margin or xwin <= margin or ywin <= margin:
            return None

        left = margin + int(np.argmax(vprof[margin:xwin]))
        right = (w - xwin) + int(np.argmax(vprof[w - xwin:w - margin]))
        top = margin + int(np.argmax(hprof[margin:ywin]))
        bottom = (h - ywin) + int(np.argmax(hprof[h - ywin:h - margin]))

        if right - left < w * 0.75 or bottom - top < h * 0.75:
            return None

        inner_pad = 2
        return (left + inner_pad, top + inner_pad,
                right - inner_pad, bottom - inner_pad)

    @staticmethod
    def _brect(mask, mf):
        h,w=mask.shape[:2]; ma=w*h*mf
        cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        best=None
        for c in cs:
            a=cv2.contourArea(c)
            if a<ma: continue
            x,y,cw,ch=cv2.boundingRect(c)
            if 1.2<=cw/max(ch,1)<=3.0:
                if best is None or a>best[0]: best=(a,x,y,cw,ch)
        return best[1:] if best else None

    # ═══════════════════════════════════════════════════════════════════════════
    # POCKETS
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_pockets(self, tb: tuple) -> list:
        x1,y1,x2,y2 = tb
        ci = max(6, self.ball_radius)
        mx = (x1+x2)//2
        self.pockets = [
            (x1+ci,y1+ci),(mx,y1),(x2-ci,y1+ci),
            (x1+ci,y2-ci),(mx,y2),(x2-ci,y2-ci),
        ]
        return self.pockets

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN  —  detect_shot  returns (ball_list, shot_dict | None)
    # ═══════════════════════════════════════════════════════════════════════════

    def detect_shot(self, frame: np.ndarray):
        if self.table_bounds is None or self._yolo is None:
            return [], None

        tx1,ty1,tx2,ty2 = self.table_bounds
        roi = frame[ty1:ty2, tx1:tx2]
        if roi.size == 0:
            return [], None

        res = self._yolo(roi, verbose=False,
                         conf=min(_BALL_CONF, _COLL_CONF),
                         device=_DEVICE, half=_HALF)[0]

        raw_balls, raw_colls = [], []

        for box in res.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            bx1,by1,bx2,by2 = map(int, box.xyxy[0])
            cx=(bx1+bx2)//2; cy=(by1+by2)//2
            cr=max(4,(bx2-bx1+by2-by1)//4)

            if cls == 0 and conf >= _BALL_CONF:
                pad=max(3,int(cr*.9))
                patch=roi[max(0,cy-pad):cy+pad, max(0,cx-pad):cx+pad]
                raw_balls.append(dict(
                    roi_xy=(cx,cy), abs_xy=(cx+tx1,cy+ty1),
                    r=cr, conf=conf, patch=patch))

            elif cls == 1 and conf >= _COLL_CONF:
                raw_colls.append(dict(
                    abs_xy=(cx+tx1,cy+ty1), r=cr, conf=conf))

        if not raw_balls:
            return [], None

        # ── update ball_radius  (EWA — α=0.15, very stable) ─────────────────
        measured = max(7, min(30, int(np.median([b["r"] for b in raw_balls]))))
        self.ball_radius = int(round(self.ball_radius * 0.85 + measured * 0.15))
        R = self.ball_radius

        # ── classify each raw detection before tracking ───────────────────────
        for b in raw_balls:
            b["ws"] = self._whiteness(b["patch"], b["r"])
        ci      = max(range(len(raw_balls)), key=lambda i: raw_balls[i]["ws"])
        has_cue = raw_balls[ci]["ws"] > 0.22

        for i, b in enumerate(raw_balls):
            if i == ci and has_cue:
                b["type"] = "cue";  b["subtype"] = "cue"
            else:
                bt = self._color(b["patch"])
                b["type"]    = bt
                b["subtype"] = bt if bt in ("cue", "8ball", "ball") else \
                               ("stripe" if self._is_stripe(b["patch"]) else "solid")

        # ── update tracks → smoothed positions ───────────────────────────────
        tracks = self._match_tracks(raw_balls)

        # ── build balls list from smoothed track positions ────────────────────
        balls   = []
        cue_pos = None
        for tr in tracks:
            ipos = (int(round(tr["pos"][0])), int(round(tr["pos"][1])))
            balls.append(dict(pos=ipos, radius=tr["r"],
                              type=tr["type"], subtype=tr["subtype"]))
            if tr["type"] == "cue":
                cue_pos = ipos

        if cue_pos is None or not self.pockets:
            return balls, None

        # ── stable ghost ball from last N frames ──────────────────────────────
        ghost_pos = self._stable_ghost(raw_colls)

        # ══ PATH A  — player is actively aiming (collision class detected) ════
        shot = None
        if ghost_pos and self.aim_conf >= _GHOST_MIN_CONF:
            aim_dir  = _norm(_v(cue_pos, ghost_pos))
            pred     = _add(ghost_pos, aim_dir, 2*R)
            target_b = min(
                (b for b in balls if b["type"] != "cue"),
                key=lambda b: _dist(b["pos"], pred),
                default=None
            )
            if target_b and _dist(target_b["pos"], pred) < R*5:
                shot = self._make_shot(cue_pos, ghost_pos,
                                       target_b, R, source="yolo")

        # ══ PATH B  — player not aiming: suggest best shot on my balls ════════
        if shot is None:
            shot = self._physics_shot(cue_pos, balls, R)

        return balls, shot

    # ── convenience alias ─────────────────────────────────────────────────────
    def detect_balls(self, frame):
        balls, _ = self.detect_shot(frame)
        return balls

    # ═══════════════════════════════════════════════════════════════════════════
    # SHOT BUILDER
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_shot(self, cue_pos, ghost_pos, target_b, R, source):
        target_pos = target_b["pos"]

        # Direction the target ball will travel after impact
        target_dir = _norm(_v(ghost_pos, target_pos))

        # Cue-ball deflection direction (90° rule, correct side)
        # Cross product sign tells us which perpendicular the cue goes toward
        aim_dir = _norm(_v(cue_pos, ghost_pos))
        cross   = _cross2d(aim_dir, target_dir)
        # Positive cross → aim hits target from right → cue deflects left of target
        if cross >= 0:
            cue_def = (-target_dir[1],  target_dir[0])  # 90° CCW
        else:
            cue_def = ( target_dir[1], -target_dir[0])  # 90° CW

        # Best pocket for target ball
        best_pocket, best_miss = None, float("inf")
        for pk in (self.pockets or []):
            # Only pockets ahead of the target ball
            t_proj = _dot(_v(target_pos, pk), target_dir)
            if t_proj < 0:
                continue
            end   = _add(target_pos, target_dir, _RAY_LEN)
            miss  = _line_dist(pk, target_pos, end)
            if miss < best_miss:
                best_miss   = miss
                best_pocket = pk

        if best_pocket is None or best_miss > _MAX_MISS:
            return None

        # Cue-ball path: starts at ghost, bounces off walls
        cue_path = self._trace(ghost_pos, cue_def, R)

        angle = self._cut_angle(cue_pos, target_pos, best_pocket)
        score = max(0.0, math.cos(math.radians(angle))) / (best_miss + 1.0)

        return dict(
            cue_pos    = cue_pos,
            ghost_pos  = ghost_pos,
            target_pos = target_pos,
            pocket     = best_pocket,
            target_dir = target_dir,
            cue_def    = cue_def,
            cue_path   = cue_path,
            ball_radius= R,
            miss_px    = best_miss,
            cut_angle  = angle,
            score      = score,
            source     = source,   # "yolo" or "physics"
        )

    # ── physics fallback: evaluate all (ball, pocket) pairs ──────────────────

    def _physics_shot(self, cue_pos, balls, R):
        best, best_sc = None, -1.0

        # When ownership is known: only suggest shots on my balls (not 8ball yet)
        if self._my_type:
            candidates = [b for b in balls
                          if b.get("subtype") == self._my_type]
        else:
            candidates = [b for b in balls
                          if b["type"] not in ("cue",)]

        # Fallback: if no mine balls visible, consider everything
        if not candidates:
            candidates = [b for b in balls if b["type"] != "cue"]

        for b in candidates:
            tpos  = b["pos"]
            aim   = _norm(_v(cue_pos, tpos))
            ghost = _add(tpos, aim, -2*R)
            shot  = self._make_shot(cue_pos, ghost, b, R, source="physics")
            if shot and shot["score"] > best_sc:
                best_sc = shot["score"]
                best    = shot
        return best

    # ═══════════════════════════════════════════════════════════════════════════
    # CUE-BALL PATH TRACER  (ball-radius-aware wall bouncing)
    # ═══════════════════════════════════════════════════════════════════════════

    def _trace(self, start, direction, R=0):
        if self.table_bounds is None:
            return [start]
        x1,y1,x2,y2 = self.table_bounds
        # The ball centre bounces at ±R from each wall
        wx1=x1+R; wy1=y1+R; wx2=x2-R; wy2=y2-R

        pts = [start]
        x,y  = float(start[0]), float(start[1])
        dx,dy = map(float, direction)

        for _ in range(_BOUNCE_STEPS):
            cands = []
            if abs(dx) > 1e-9:
                t = (wx2-x)/dx if dx>0 else (wx1-x)/dx
                cands.append((t,"x"))
            if abs(dy) > 1e-9:
                t = (wy2-y)/dy if dy>0 else (wy1-y)/dy
                cands.append((t,"y"))
            if not cands:
                break
            t_min, ax = min(cands, key=lambda c: c[0] if c[0]>1e-3 else 1e9)
            if t_min <= 1e-3:
                break
            x += dx*t_min;  y += dy*t_min
            pts.append((x, y))
            if ax == "x": dx = -dx
            else:         dy = -dy

        return pts

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _whiteness(self, patch, cr):
        if patch is None or patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        r   = max(3, int(cr*.85))
        h,w = hsv.shape[:2]
        cy,cx = h//2, w//2
        p = hsv[max(0,cy-r):cy+r, max(0,cx-r):cx+r]
        if p.size == 0:
            return 0.0
        v,s = p[:,:,2], p[:,:,1]
        n   = p.shape[0]*p.shape[1]
        return float(np.sum((v>170)&(s<70))) / max(n,1) * (float(np.mean(v))/255)

    def _is_stripe(self, patch) -> bool:
        """True when ball has a prominent white band across the middle rows."""
        if patch is None or patch.size == 0:
            return False
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hh, ww = hsv.shape[:2]
        y0, y1 = int(hh * .33), int(hh * .67)
        mid = hsv[y0:y1, :]
        if mid.size == 0:
            return False
        mv = mid[:,:,2].flatten();  ms = mid[:,:,1].flatten()
        v  = hsv[:,:,2].flatten();  s  = hsv[:,:,1].flatten()
        mid_white   = np.sum((mv > 165) & (ms < 75)) / max(len(mv), 1)
        total_white = np.sum((v  > 165) & (s  < 75)) / max(len(v),  1)
        return mid_white > 0.22 and total_white < 0.42

    @staticmethod
    def _cut_angle(cue, target, pocket):
        a1 = _norm(_v(cue, target))
        a2 = _norm(_v(target, pocket))
        return math.degrees(math.acos(max(-1.0, min(1.0, _dot(a1,a2)))))

    # ── colour classifier ─────────────────────────────────────────────────────
    _CR = {
        "yellow":([15,100,100],[35,255,255]),  "blue":([90,100,70],[130,255,255]),
        "red":   ([0, 110, 70],[12, 255,255]), "red2":([163,110,65],[180,255,255]),
        "purple":([120,55, 45],[165,255,255]), "orange":([10,130,95],[22, 255,255]),
        "green": ([35, 85, 45],[87, 255,255]), "maroon":([6,  40, 28],[18, 150,120]),
    }

    def _color(self, patch):
        if patch is None or patch.size == 0:
            return "ball"
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0].flatten().astype(float)
        s = hsv[:,:,1].flatten().astype(float)
        v = hsv[:,:,2].flatten().astype(float)
        n = len(h)
        if np.sum((v>170)&(s<70))/n > 0.35: return "cue"
        if np.sum((v<65)&(s<80))/n > 0.55:  return "8ball"
        sat = (s>80)&(v>60)
        if sat.sum() < max(4, int(n*.05)): return "ball"
        hm=float(np.median(h[sat])); sm=float(np.median(s[sat]))
        vm=float(np.median(v[sat]))
        for nm,(lo,hi) in self._CR.items():
            if lo[0]<=hm<=hi[0] and sm>=lo[1] and vm>=lo[2]:
                return nm
        return "ball"
