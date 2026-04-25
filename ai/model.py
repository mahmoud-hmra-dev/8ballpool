"""
Shot prediction model.

Input  : board state (cue pos, ball positions, my_type)
Output : ghost_pos (x, y) normalized [0,1]

Architecture: MLP with residual connections
"""
import torch
import torch.nn as nn

MAX_BALLS = 15          # max non-cue balls (padded with zeros)
BALL_FEAT  = 4          # x, y, is_solid, is_stripe
CUE_FEAT   = 2          # x, y
TYPE_FEAT  = 3          # one-hot: unknown, solid, stripe
INPUT_DIM  = MAX_BALLS * BALL_FEAT + CUE_FEAT + TYPE_FEAT   # 65


class ShotNet(nn.Module):

    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),

            nn.Linear(hidden // 2, 2),
            nn.Sigmoid(),          # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── feature encoding ──────────────────────────────────────────────────────────

def encode(record: dict) -> list[float]:
    """
    Convert one dataset record → flat float vector of length INPUT_DIM.
    """
    # cue
    cx, cy = record["cue_n"]
    feats = [cx, cy]

    # my_type one-hot
    t = record.get("my_type", "unknown")
    feats += [
        1.0 if t == "unknown" else 0.0,
        1.0 if t == "solid"   else 0.0,
        1.0 if t == "stripe"  else 0.0,
    ]

    # balls (sorted by x for determinism, padded to MAX_BALLS)
    balls = sorted(record["balls"], key=lambda b: b["pos_n"][0])
    for b in balls[:MAX_BALLS]:
        bx, by = b["pos_n"]
        sub = b.get("subtype", "")
        feats += [
            bx, by,
            1.0 if sub == "solid"  else 0.0,
            1.0 if sub == "stripe" else 0.0,
        ]
    # pad missing balls with zeros
    pad = MAX_BALLS - min(len(balls), MAX_BALLS)
    feats += [0.0] * (pad * BALL_FEAT)

    return feats


def augment(record: dict) -> list[dict]:
    """
    Return 8 augmented variants of one record:
      - horizontal flip   (x → 1-x)
      - vertical flip     (y → 1-y)
      - both flips
      - each × small noise
    """
    def flip(r, fx, fy):
        def fp(pos): return [1-pos[0] if fx else pos[0],
                             1-pos[1] if fy else pos[1]]
        return {
            "my_type": r["my_type"],
            "cue_n":   fp(r["cue_n"]),
            "ghost_n": fp(r["ghost_n"]),
            "balls":   [{"pos_n": fp(b["pos_n"]),
                         "type": b["type"], "subtype": b["subtype"]}
                        for b in r["balls"]],
        }

    import random
    def noisy(r, sigma=0.015):
        def np_(pos): return [max(0, min(1, v + random.gauss(0, sigma)))
                              for v in pos]
        return {
            "my_type": r["my_type"],
            "cue_n":   np_(r["cue_n"]),
            "ghost_n": np_(r["ghost_n"]),
            "balls":   [{"pos_n": np_(b["pos_n"]),
                         "type": b["type"], "subtype": b["subtype"]}
                        for b in r["balls"]],
        }

    variants = [
        flip(record, False, False),   # original
        flip(record, True,  False),   # h-flip
        flip(record, False, True),    # v-flip
        flip(record, True,  True),    # both
    ]
    # add noisy copies
    return variants + [noisy(v) for v in variants]
