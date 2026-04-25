"""
YOLO inference — synchronous wrapper + async background runner.

YOLOInference   — thin stateless wrapper; each run() call blocks until done.
AsyncYOLOInference — runs YOLOInference in a dedicated thread so the main
                     pipeline loop is never blocked waiting for the GPU.

Async design
------------
  submit(roi, offset)  — non-blocking; queues the frame for the worker thread.
                         If the worker is still busy, the waiting frame is
                         replaced (we always want the freshest frame, not a backlog).
  latest_result()      — returns (raw_balls, raw_colls) from the last completed run.

Result lag is exactly one YOLO cycle (~55 ms at 18 FPS).
The EWA tracker absorbs this gracefully — ball positions are already smoothed
across frames, so a one-frame lag is invisible to the user.

YOLO classes
  0  ball       → every pool ball on the table
  1  collision  → ghost ball (where cue centre must be at impact)
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

try:
    import torch
    _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    _HALF   = torch.cuda.is_available()
except ImportError:
    _DEVICE = "cpu"
    _HALF   = False

from config import BALL_CONF, COLL_CONF, YOLO_IMGSZ

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE_PATH = os.path.join(_ROOT, "best.engine")
_PT_PATH     = os.path.join(_ROOT, "best.pt")
_YOLO_PATH   = _ENGINE_PATH if os.path.exists(_ENGINE_PATH) else _PT_PATH


# ── Synchronous wrapper ───────────────────────────────────────────────────────

class YOLOInference:
    """
    Thin, stateless YOLO wrapper.
    Each call to run() is independent — no state carried between frames.
    """

    def __init__(self):
        self._model = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(_ENGINE_PATH):
            print(
                "[Inference] TRT engine not found — using best.pt\n"
                "  For 3-5× speedup, run once:\n"
                f"  python -c \"from ultralytics import YOLO; "
                f"YOLO(r'{_PT_PATH}').export(format='engine', half=True, device=0)\""
            )
        try:
            from ultralytics import YOLO
            self._model = YOLO(_YOLO_PATH)
            self._model(
                np.zeros((64, 64, 3), np.uint8),
                verbose=False, device=_DEVICE, half=_HALF, imgsz=YOLO_IMGSZ,
            )
            tag = "TRT" if _YOLO_PATH.endswith(".engine") else "PT"
            print(f"[Inference] YOLO ready [{tag}]  device={_DEVICE}  "
                  f"half={_HALF}  imgsz={YOLO_IMGSZ}")
        except Exception as e:
            print(f"[Inference] Failed to load model: {e}")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def run(self, roi: np.ndarray, offset: tuple[int, int]) -> tuple[list, list]:
        """
        Run YOLO on a table ROI (blocking).

        offset = (tx1, ty1) — top-left of the ROI in the full frame,
                 converts ROI-local coords → frame-absolute coords.

        Returns raw_balls, raw_colls as plain dicts.
        """
        if not self.ready or roi.size == 0:
            return [], []

        tx, ty  = offset
        results = self._model(
            roi, verbose=False,
            conf=min(BALL_CONF, COLL_CONF),
            device=_DEVICE, half=_HALF, imgsz=YOLO_IMGSZ,
        )[0]

        raw_balls: list[dict] = []
        raw_colls: list[dict] = []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            cr = max(4, (bx2 - bx1 + by2 - by1) // 4)

            if cls == 0 and conf >= BALL_CONF:
                pad   = max(3, int(cr * 0.9))
                patch = roi[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
                raw_balls.append({
                    "roi_xy": (cx, cy),
                    "abs_xy": (cx + tx, cy + ty),
                    "r": cr, "conf": conf, "patch": patch,
                })
            elif cls == 1 and conf >= COLL_CONF:
                raw_colls.append({
                    "abs_xy": (cx + tx, cy + ty),
                    "r": cr, "conf": conf,
                })

        return raw_balls, raw_colls


# ── Async wrapper ─────────────────────────────────────────────────────────────

class AsyncYOLOInference:
    """
    Runs YOLOInference in a dedicated background thread.

    The main pipeline loop calls submit() every frame (non-blocking) and reads
    the last completed result via latest_result(). The GPU is always busy, and
    the CPU-side loop is never stalled waiting for inference.

    Input queue size = 1.  If YOLO is still running when a new frame arrives,
    the pending frame is replaced — we prefer fresh data over a growing backlog.
    """

    def __init__(self):
        self._sync = YOLOInference()

        # Shared state — protected by their respective locks
        self._pending:        Optional[tuple] = None   # (roi, offset)
        self._result:         tuple           = ([], [])
        self._pending_lock    = threading.Lock()
        self._result_lock     = threading.Lock()
        self._new_frame       = threading.Event()

        t = threading.Thread(target=self._worker, daemon=True, name="yolo-worker")
        t.start()

    @property
    def ready(self) -> bool:
        return self._sync.ready

    def submit(self, roi: np.ndarray, offset: tuple[int, int]) -> None:
        """
        Queue a frame for YOLO inference (non-blocking).
        .copy() is mandatory: the main loop may overwrite the array before
        the worker thread gets to read it.
        """
        with self._pending_lock:
            self._pending = (roi.copy(), offset)
        self._new_frame.set()

    def latest_result(self) -> tuple[list, list]:
        """Return (raw_balls, raw_colls) from the last completed inference."""
        with self._result_lock:
            return self._result

    def _worker(self) -> None:
        while True:
            # Sleep until a new frame is submitted
            self._new_frame.wait()
            self._new_frame.clear()

            with self._pending_lock:
                if self._pending is None:
                    continue
                roi, offset    = self._pending
                self._pending  = None

            balls, colls = self._sync.run(roi, offset)

            with self._result_lock:
                self._result = (balls, colls)
