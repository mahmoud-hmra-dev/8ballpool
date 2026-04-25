"""
Table detector.

Locates the pool table felt bounds within a captured frame.
Returns a TableBounds dataclass, or None on failure.

Detection pipeline (three strategies, applied in order of reliability):
  1. _felt   — direct HSV color match on the green/teal felt
  2. _sat    — broad saturation mask (fallback for unusual felt colors)
  3. _edges  — Canny edge detection (last resort)

All three strategies work on a 50%-scaled image for speed.
A two-pass refinement (_refine + _inner_edges) tightens the final bounds
using Sobel gradients to find the actual playing surface edges.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import SKIP_TOP_FRAC
from models import TableBounds


class TableDetector:

    def detect(self, frame: np.ndarray) -> TableBounds | None:
        """
        Detect the table in a full captured frame.
        Returns TableBounds (with ball_radius estimated from table size),
        or None if the table cannot be found.
        """
        fh, fw = frame.shape[:2]
        skip   = int(fh * SKIP_TOP_FRAC)   # skip score panel at top
        search = frame[skip:, :]

        # Work at 50% scale for the coarse search — fast and sufficient
        sc  = 0.5
        sm  = cv2.resize(search, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)

        outer = self._felt(hsv) or self._sat(hsv) or self._edges(sm)

        if outer is None:
            # Hard-coded fallback: assume table occupies the central area
            ox1 = int(fw * .30); ox2 = int(fw * .85)
            oy1 = skip + int((fh - skip) * .05); oy2 = int(fh * .90)
        else:
            sx, sy, sw, sh = outer
            ox1 = int(sx / sc);         oy1 = int(sy / sc) + skip
            ox2 = int((sx + sw) / sc);  oy2 = int((sy + sh) / sc) + skip

        ox1 = max(0, ox1); oy1 = max(0, oy1)
        ox2 = min(fw, ox2); oy2 = min(fh, oy2)

        # Refine bounds at full scale
        r = self._refine(frame[oy1:oy2, ox1:ox2])
        if r:
            fx, fy, fw2, fh2 = r
            x1, y1 = ox1 + fx, oy1 + fy
            x2, y2 = x1 + fw2, y1 + fh2
        else:
            tw, th = ox2 - ox1, oy2 - oy1
            x1 = ox1 + max(8, int(tw * .07)); y1 = oy1 + max(8, int(th * .10))
            x2 = ox2 - max(8, int(tw * .07)); y2 = oy2 - max(8, int(th * .10))

        # Fine-tune with gradient-based inner edge detection
        inner = self._inner_edges(frame[y1:y2, x1:x2])
        if inner is not None:
            ix1, iy1, ix2, iy2 = inner
            x1 += ix1; y1 += iy1
            x2  = x1 + (ix2 - ix1); y2 = y1 + (iy2 - iy1)

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)
        tw, th = x2 - x1, y2 - y1

        ball_r = max(7, min(30, max(int(tw / 96), int(th / 48))))
        inset  = max(2, int(round(ball_r * 0.15)))
        return TableBounds(
            x1=x1 + inset, y1=y1 + inset,
            x2=x2 - inset, y2=y2 - inset,
            ball_radius=ball_r,
        )

    # ── detection strategies (work on half-scale HSV / BGR) ───────────────────

    def _felt(self, hsv: np.ndarray):
        """HSV color match on standard green/teal billiard felt."""
        m  = cv2.inRange(hsv, np.array([78, 50, 65]),  np.array([118, 210, 225]))
        m |= cv2.inRange(hsv, np.array([55, 45, 55]),  np.array([80,  195, 205]))
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        return self._largest_rect(
            cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k),
            min_frac=0.08)

    def _sat(self, hsv: np.ndarray):
        """Broad saturation-based fallback for unusual felt colors."""
        m = cv2.inRange(hsv, np.array([0, 40, 35]), np.array([180, 255, 215]))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        return self._largest_rect(
            cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k),
            min_frac=0.12)

    def _edges(self, sm: np.ndarray):
        """Canny edge detection — last resort when color fails."""
        g = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 1), 25, 80)
        g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        return self._largest_rect(g, min_frac=0.10)

    # ── refinement (work on full-scale ROI) ───────────────────────────────────

    def _refine(self, roi: np.ndarray):
        """Re-run felt detection on the coarse ROI at 50% scale for tighter bounds."""
        if roi.size == 0:
            return None
        sc = 0.5
        sm = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        h  = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        m  = cv2.inRange(h, np.array([78, 50, 65]),  np.array([118, 210, 225]))
        m |= cv2.inRange(h, np.array([55, 45, 55]),  np.array([80,  195, 205]))
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        r  = self._largest_rect(
            cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k),
            min_frac=0.06)
        if r is None:
            return None
        x, y, w, hh = r
        return (int(x / sc), int(y / sc), int(w / sc), int(hh / sc))

    def _inner_edges(self, roi: np.ndarray):
        """
        Use Sobel gradients to find the actual playing-surface edges.
        Returns (x1, y1, x2, y2) relative to the roi, or None.
        """
        if roi.size == 0:
            return None
        h, w = roi.shape[:2]
        if w < 80 or h < 60:
            return None

        blur = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gx   = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        gy   = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))

        by1, by2 = int(h * .18), int(h * .82)
        bx1, bx2 = int(w * .18), int(w * .82)
        if by2 <= by1 or bx2 <= bx1:
            return None

        vprof = cv2.GaussianBlur(
            gx[by1:by2].mean(axis=0).astype(np.float32).reshape(1, -1),
            (1, 31), 0).ravel()
        hprof = cv2.GaussianBlur(
            gy[:, bx1:bx2].mean(axis=1).astype(np.float32).reshape(-1, 1),
            (31, 1), 0).ravel()

        margin = 8
        xwin   = max(24, int(w * .12))
        ywin   = max(24, int(h * .12))
        if w <= 2 * margin or h <= 2 * margin or xwin <= margin or ywin <= margin:
            return None

        left   = margin + int(np.argmax(vprof[margin:xwin]))
        right  = (w - xwin) + int(np.argmax(vprof[w - xwin:w - margin]))
        top    = margin + int(np.argmax(hprof[margin:ywin]))
        bottom = (h - ywin) + int(np.argmax(hprof[h - ywin:h - margin]))

        if right - left < w * 0.75 or bottom - top < h * 0.75:
            return None

        pad = 2
        return (left + pad, top + pad, right - pad, bottom - pad)

    # ── shared helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _largest_rect(mask: np.ndarray, min_frac: float):
        """
        Find the largest rectangular contour that looks like a pool table
        (aspect ratio 1.2–3.0) and covers at least min_frac of the image.
        Returns (x, y, w, h) or None.
        """
        h, w  = mask.shape[:2]
        min_a = w * h * min_frac
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best  = None
        for c in cs:
            a = cv2.contourArea(c)
            if a < min_a:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if 1.2 <= cw / max(ch, 1) <= 3.0:
                if best is None or a > best[0]:
                    best = (a, x, y, cw, ch)
        return best[1:] if best else None
