"""
YOLO inference wrapper.

Single responsibility: run the model on a frame ROI and return raw detections.
All temporal logic (tracking, smoothing) lives in tracker.py — not here.

YOLO classes
  0  ball       → every pool ball on the table
  1  collision  → ghost ball (where cue centre must be at impact)

Auto-selects TensorRT engine if best.engine exists (3-5× faster than .pt).
"""
import os

import numpy as np

try:
    import torch
    _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    _HALF   = torch.cuda.is_available()
except ImportError:
    _DEVICE = "cpu"
    _HALF   = False

from config import BALL_CONF, COLL_CONF

# Model is at the project root (one level above this file)
_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE_PATH  = os.path.join(_ROOT, "best.engine")
_PT_PATH      = os.path.join(_ROOT, "best.pt")
_YOLO_PATH    = _ENGINE_PATH if os.path.exists(_ENGINE_PATH) else _PT_PATH


class YOLOInference:
    """
    Thin, stateless YOLO wrapper.
    Stateless means: each call to run() is independent — no memory between frames.
    """

    def __init__(self):
        self._model = None
        self._load()

    def _load(self):
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
            # Warm-up: compiles CUDA kernels so the first real frame isn't slow
            self._model(np.zeros((64, 64, 3), np.uint8),
                        verbose=False, device=_DEVICE, half=_HALF)
            tag = "TRT" if _YOLO_PATH.endswith(".engine") else "PT"
            print(f"[Inference] YOLO ready [{tag}]  device={_DEVICE}  half={_HALF}")
        except Exception as e:
            print(f"[Inference] Failed to load model: {e}")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def run(
        self,
        roi:    np.ndarray,
        offset: tuple[int, int],
    ) -> tuple[list[dict], list[dict]]:
        """
        Run YOLO on a table ROI.

        Parameters
        ----------
        roi    : BGR frame cropped to the table bounds
        offset : (tx1, ty1) — top-left of the ROI in the full frame,
                 used to convert ROI-local coords → frame-absolute coords

        Returns
        -------
        raw_balls : [{roi_xy, abs_xy, r, conf, patch}, ...]
        raw_colls : [{abs_xy, r, conf}, ...]
        """
        if not self.ready or roi.size == 0:
            return [], []

        tx, ty = offset
        results = self._model(
            roi, verbose=False,
            conf=min(BALL_CONF, COLL_CONF),
            device=_DEVICE, half=_HALF,
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
