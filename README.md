# 8 Ball Pool — AI Shot Assistant & Auto-Player

Real-time AI overlay for 8 Ball Pool. Detects balls via YOLO, calculates the best shot using physics geometry, draws aim lines on a transparent overlay, and can **automatically play** by controlling the aim wheel and power bar — all at 60+ FPS.

Supports two capture sources: **Chrome browser** (desktop game) and **scrcpy** (mobile game mirrored over USB/WiFi).

---

## Requirements

- Windows 10/11
- Python 3.12+
- NVIDIA GPU (recommended — CPU works but slower)
- [scrcpy](https://github.com/Genymobile/scrcpy) *(only for mobile/auto-play mode)*

---

## Installation

```bash
git clone https://github.com/mahmoud-hmra-dev/8ballpool.git
cd 8ballpool
pip install -r requirements.txt
```

**GPU support (recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Optional — 3–5× faster YOLO inference (TensorRT):**
```bash
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='engine', half=True, device=0)"
```
Run once. The engine file is loaded automatically on the next startup.

---

## Quick Start

### Chrome (desktop game — overlay only)

```bash
python main.py --source chrome
```

### scrcpy (mobile game — overlay only)

1. Connect your phone via USB with USB debugging enabled
2. Start scrcpy: `scrcpy`
3. Open the game on your phone
4. Run:
```bash
python main.py --source scrcpy
```

### Auto-play (AI controls the game automatically)

```bash
python main.py --source scrcpy --autoplay
```

The AI will:
1. Detect the current ball positions
2. Predict the best shot using the trained model
3. Rotate the aim wheel to the correct angle
4. Pull the power bar and release to shoot

> **Note:** Requires UI calibration and a trained model (see sections below).

---

## Table Calibration

If the auto-detected table rectangle is wrong, calibrate it manually:

```bash
python calibrate.py --source scrcpy
```

A window opens showing the live capture. Draw a rectangle over the table:

| Key | Action |
|-----|--------|
| **Drag** | Draw rectangle |
| **Enter** | Save and exit |
| **R** | Reset rectangle |
| **Esc** | Cancel |

Saved to `calibration/table_scrcpy.json` — loaded automatically on every future run.

---

## UI Calibration (Auto-Play)

Before using `--autoplay`, you must calibrate the aim wheel and power bar positions once:

```bash
python ai/game_controller.py --calibrate
```

A window opens showing your scrcpy screen. Click in this order:

1. **Center of the aim wheel** (right side of screen)
2. **Top of the power bar / cue stick** (left side of screen — click the very top for full power)

Saved to `calibration/ui_scrcpy.json`.

> **Tip:** If the aim angle is slightly off, adjust `px_per_deg` in `calibration/ui_scrcpy.json`.
> Higher value = smaller wheel drag per degree. Default: `4.0`.

---

## Data Collection & AI Training

### Step 1 — Collect shots while you play

```bash
python main.py --source scrcpy --collect
```

Every shot you aim is recorded to `ai/dataset/shots_YYYY-MM-DD.jsonl`.

### Step 2 — Train the model

After collecting enough shots (300+ recommended):

```bash
python ai/train.py
```

The trained model is saved to `ai/shot_model.pt`.

| Shots collected | Expected accuracy |
|-----------------|-------------------|
| ~30 (starter)   | ~23% error |
| ~200            | ~12% error |
| ~500            | ~6% error  |
| 1000+           | ~3% error  |

Re-run `python ai/train.py` after each session to retrain on the latest data.

---

## Touch Test

Verify that wheel rotation and shooting work correctly before running auto-play:

```bash
python test_touch.py
```

This script:
1. Shows the screen coordinates for the wheel and power bar
2. Moves the cursor to each position and takes a screenshot (`test_wheel.png`, `test_power.png`)
3. Performs a wheel swipe (aim rotation)
4. Performs a power bar drag (full-power shot)

---

## Power Levels

The auto-player uses three power levels (configurable in `ai/game_controller.py`):

| Level | Drag distance | Usage |
|-------|--------------|-------|
| `light` | 200 px | Short shots |
| `medium` | 400 px | Normal shots |
| `hard` | 620 px | Full power |

---

## Project Structure

```
8ballpool/
├── main.py                  # Entry point & pipeline orchestrator
├── calibrate.py             # Interactive table calibration tool
├── test_touch.py            # Touch/wheel/power bar test script
├── config.py                # All tunable constants
├── models.py                # Data classes: Ball, TableBounds, Shot
├── requirements.txt
│
├── pipeline/
│   ├── capture.py           # Win32 screen capture (async)
│   ├── inference.py         # YOLO ball & ghost-ball detection (async GPU thread)
│   ├── classifier.py        # HSV-based ball type/color classification
│   ├── tracker.py           # EWA ball tracking + ghost-ball buffer
│   ├── shot_engine.py       # Physics: cut angle, ghost-ball, bounce tracing
│   └── table_detector.py    # Table bounds detection (HSV → Sobel → fallback)
│
├── overlay/
│   ├── window.py            # Transparent click-through overlay
│   └── renderer.py          # Shot visualization (aim line, paths, HUD)
│
├── ai/
│   ├── data_collector.py    # Records shots during gameplay
│   ├── model.py             # ShotNet neural network (MLP, 65-dim input)
│   ├── predictor.py         # Loads model and runs inference
│   ├── train.py             # Training script (8× augmentation, 600 epochs)
│   ├── auto_player.py       # Executes shots via ScrcpyTouch
│   ├── game_controller.py   # Aim wheel + power bar control logic
│   ├── scrcpy_touch.py      # Win32 mouse injection (SetCursorPos + mouse_event)
│   └── dataset/             # Recorded shots (auto-created)
│
└── calibration/             # Saved calibration files (auto-created)
    ├── table_scrcpy.json    # Table bounds
    ├── table_chrome.json
    └── ui_scrcpy.json       # Wheel + power bar positions
```

---

## How It Works

```
Screen capture (Win32 / mss)
        ↓
YOLO inference — detects balls & ghost ball (async GPU thread)
        ↓
Ball classification — cue / solid / stripe / 8-ball (HSV)
        ↓
Ball tracking — EWA smoothing, ghost-ball stabilization
        ↓
Shot selection:
  Overlay mode  → follow ghost ball or suggest best physics shot
  Auto-play     → AI model predicts target → aim wheel + power bar
        ↓
Overlay — aim line, trajectory, pocket targets (60+ FPS)
Auto-play — SetCursorPos + mouse_event → scrcpy → phone touch
```

**Touch injection:** Uses `SetCursorPos` + `win32api.mouse_event` which goes through the Windows system input pipeline — more reliable for SDL (scrcpy) windows than PostMessage. The window is brought to foreground automatically before each interaction.

**Coordinate scaling:** UI calibration coordinates are saved with the frame size at calibration time. If the scrcpy window is resized, coordinates are scaled automatically at runtime.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| scrcpy window not found | Make sure scrcpy is open (not the terminal, the SDL display window) |
| Table rectangle wrong | Run `python calibrate.py --source scrcpy` |
| No balls detected | Lower `BALL_CONF` in `config.py` (try `0.20`) |
| Aim angle off | Adjust `px_per_deg` in `calibration/ui_scrcpy.json` |
| Shot too weak | Increase `hard` value in `POWER_DRAG_PX` in `ai/game_controller.py` |
| Auto-play fires too fast | Increase `POST_SHOT_WAIT` in `_run_autoplay` in `main.py` |
| Low FPS | Export TensorRT engine (see Installation) |
| Overlay not visible | Run as administrator |
