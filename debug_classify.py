"""
Diagnostic: show ball classification on debug_frame.png
Run: python debug_classify.py
"""
import cv2
import numpy as np
import json
import os

img = cv2.imread("debug_frame.png")
if img is None:
    print("Run main.py first to generate debug_frame.png")
    exit()

# Load calibration to know table bounds
calib = None
for src in ("scrcpy", "chrome"):
    p = f"calibration/table_{src}.json"
    if os.path.exists(p):
        with open(p) as f:
            calib = json.load(f)
        break

COLOR_MAP = {
    "cue":    (255, 255, 255),
    "8ball":  (0,   0,   0  ),
    "solid":  (0,   200, 255),
    "stripe": (0,   165, 255),
    "ball":   (128, 128, 128),
}

LABEL_MAP = {
    "cue":    "CUE",
    "8ball":  "8",
    "solid":  "S",
    "stripe": "T",
    "ball":   "?",
}

# Run YOLO on the table ROI
from pipeline.classifier import classify_ball, whiteness_score, classify_color
from pipeline.inference import AsyncYOLOInference
import time

yolo = AsyncYOLOInference()

if calib:
    x1, y1, x2, y2 = calib["x1"], calib["y1"], calib["x2"], calib["y2"]
    roi = img[y1:y2, x1:x2]
    offset = (x1, y1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
else:
    roi = img
    offset = (0, 0)

yolo.submit(roi, offset=offset)
time.sleep(1.0)
balls, _ = yolo.latest_result()

if not balls:
    print("No balls detected — check YOLO model")
    exit()

# Compute whiteness scores
for b in balls:
    b["ws"] = whiteness_score(b["patch"], b["r"])

cue_idx = max(range(len(balls)), key=lambda i: balls[i]["ws"])
has_cue = balls[cue_idx]["ws"] > 0.22

counts = {"cue": 0, "8ball": 0, "solid": 0, "stripe": 0, "ball": 0}

for i, b in enumerate(balls):
    if i == cue_idx and has_cue:
        t, sub = "cue", "cue"
    else:
        t, sub = classify_ball(b["patch"], b["r"])
        if t == "cue":
            t, sub = "ball", "ball"

    counts[t] = counts.get(t, 0) + 1

    px, py = b["abs_xy"]
    r = b["r"]
    color = COLOR_MAP.get(sub if sub in COLOR_MAP else t, (128, 128, 128))
    label = LABEL_MAP.get(sub if sub in LABEL_MAP else t, "?")

    cv2.circle(img, (px, py), r + 3, color, 2)
    cv2.putText(img, label, (px - 8, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    # show whiteness score for debugging
    cv2.putText(img, f"{b['ws']:.2f}", (px - 12, py + r + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

# Summary
print(f"\nDetected: {len(balls)} balls")
for k, v in counts.items():
    print(f"  {k:8s}: {v}")

cv2.putText(img, f"CUE:{counts['cue']}  8:{counts['8ball']}  S:{counts['solid']}  T:{counts['stripe']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

out = "debug_classify.png"
cv2.imwrite(out, img)
print(f"\nSaved: {out}")
print("S=solid  T=stripe  8=8ball  CUE=cue  ?=unknown")

cv2.imshow("Ball Classification", cv2.resize(img, (1280, 720)))
cv2.waitKey(0)
cv2.destroyAllWindows()
