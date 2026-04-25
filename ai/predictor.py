"""
ShotPredictor — loads the trained model and predicts ghost_pos.

Usage:
    pred = ShotPredictor("ai/shot_model.pt")
    ghost_frame = pred.predict(balls, cue_pos, my_type, table)
    # returns (x, y) in frame coordinates, or None if model not ready
"""
import os
from typing import Optional

import torch

from ai.model import ShotNet, encode

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "shot_model.pt")


class ShotPredictor:

    def __init__(self, model_path: str = _MODEL_PATH):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model  = ShotNet().to(self._device)
        self._ready  = False

        if not os.path.exists(model_path):
            print(f"[ShotPredictor] model not found at {model_path}")
            return

        ckpt = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._ready = True
        val = ckpt.get("val_loss", "?")
        print(f"[ShotPredictor] model loaded — val_loss={val:.5f}  device={self._device}")

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self,
                balls:   list,
                cue_pos: tuple,
                my_type: Optional[str],
                table) -> Optional[tuple[float, float]]:
        """
        Returns (ghost_x, ghost_y) in FRAME coordinates, or None.
        """
        if not self._ready or cue_pos is None:
            return None

        tw = table.x2 - table.x1
        th = table.y2 - table.y1
        if tw <= 0 or th <= 0:
            return None

        record = {
            "my_type": my_type or "unknown",
            "cue_n": [
                round((cue_pos[0] - table.x1) / tw, 4),
                round((cue_pos[1] - table.y1) / th, 4),
            ],
            "ghost_n": [0, 0],   # unused during inference
            "balls": [
                {
                    "pos_n": [
                        round((b["pos"][0] - table.x1) / tw, 4),
                        round((b["pos"][1] - table.y1) / th, 4),
                    ],
                    "type":    b["type"],
                    "subtype": b.get("subtype", ""),
                }
                for b in balls if b["type"] != "cue"
            ],
        }

        x_in = torch.tensor([encode(record)], dtype=torch.float32).to(self._device)
        with torch.no_grad():
            gx_n, gy_n = self._model(x_in)[0].cpu().tolist()

        # denormalize → frame coordinates
        ghost_x = table.x1 + gx_n * tw
        ghost_y = table.y1 + gy_n * th
        return (ghost_x, ghost_y)
